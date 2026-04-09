from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from typing import Any

import numpy as np
import tensorflow as tf
from matplotlib import cm
from PIL import Image
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.segmentation import find_boundaries, slic
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

from .state import ExplainabilityResult

try:
    from lime import lime_image

    LIME_AVAILABLE = True
except ImportError:
    lime_image = None
    LIME_AVAILABLE = False


PIXEL_SPACING_MM = 0.5
SLICE_THICKNESS_MM = 5.0


def _to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    encoded = b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype('float32'), nan=0.0, posinf=0.0, neginf=0.0)
    values -= float(values.min())
    max_value = float(values.max())
    if max_value <= 1e-8:
        return np.zeros_like(values, dtype='float32')
    return values / max_value


def _find_last_conv_layer(model) -> str:
    last_name: str | None = None

    def walk(layer):
        nonlocal last_name
        if isinstance(layer, Conv2D):
            last_name = layer.name
        if hasattr(layer, 'layers'):
            for child in layer.layers:
                walk(child)

    walk(model)
    if not last_name:
        raise ValueError('No convolutional layer found for Grad-CAM computation.')
    return last_name


def _get_nested_layer(model, target_name: str):
    for layer in model.layers:
        if layer.name == target_name:
            return layer
        if hasattr(layer, 'layers'):
            try:
                return layer.get_layer(target_name)
            except Exception:
                nested = _get_nested_layer(layer, target_name)
                if nested is not None:
                    return nested
    return None



def _build_vgg16_grad_models(model, last_conv_layer_name: str) -> tuple[Model, Model]:
    base_model = model.get_layer('vgg16')
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = Model(base_model.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    use_layers = False
    for layer in base_model.layers:
        if layer.name == last_conv_layer_name:
            use_layers = True
            continue
        if use_layers:
            x = layer(x)

    for layer in model.layers[1:]:
        x = layer(x)

    classifier_model = Model(classifier_input, x)
    return conv_model, classifier_model

def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled = label(mask.astype(np.uint8))
    if labeled.max() == 0:
        return mask.astype(np.uint8)
    largest = max(regionprops(labeled), key=lambda region: region.area)
    return (labeled == largest.label).astype(np.uint8)


def _clean_mask(mask: np.ndarray, min_pixels: int = 24) -> np.ndarray:
    cleaned = binary_opening(mask.astype(bool), footprint=disk(1))
    cleaned = binary_closing(cleaned, footprint=disk(2))
    cleaned = remove_small_objects(cleaned, min_size=min_pixels)
    if np.count_nonzero(cleaned) == 0:
        return mask.astype(np.uint8)
    return _largest_component(cleaned).astype(np.uint8)


def _build_mask_from_heatmap(heatmap: np.ndarray, quantile: float = 0.82) -> np.ndarray:
    normalized = _normalize_map(heatmap)
    positive = normalized[normalized > 0]
    if positive.size == 0:
        return np.zeros_like(normalized, dtype=np.uint8)

    otsu_value = float(threshold_otsu(positive)) if positive.size > 8 else 0.0
    quantile_value = float(np.quantile(positive, quantile))
    threshold = max(otsu_value, quantile_value * 0.85)
    mask = normalized >= threshold
    if np.count_nonzero(mask) < 20:
        threshold = float(np.quantile(positive, 0.90))
        mask = normalized >= threshold
    return _clean_mask(mask)


def _compute_mask_metrics(mask: np.ndarray, heatmap: np.ndarray) -> tuple[int, float, float, float]:
    area_px = int(np.count_nonzero(mask))
    total_px = int(mask.size)
    area_ratio = float(area_px / total_px) if total_px else 0.0
    if area_px == 0:
        return 0, 0.0, 0.0, 0.0

    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox_area_ratio = float(((y_max - y_min + 1) * (x_max - x_min + 1)) / total_px)
    dispersion = float((np.std(coords[:, 0]) / mask.shape[0] + np.std(coords[:, 1]) / mask.shape[1]) / 2.0)
    spread_score = float(min(1.0, bbox_area_ratio * 0.60 + dispersion * 0.40))
    activation = _normalize_map(heatmap)
    masked_intensity = float(np.mean(activation[mask > 0]))
    return area_px, area_ratio, spread_score, masked_intensity


def _estimate_volume_cm3(area_px: int) -> float:
    area_mm2 = area_px * (PIXEL_SPACING_MM ** 2)
    volume_mm3 = area_mm2 * SLICE_THICKNESS_MM
    return float(volume_mm3 / 1000.0)


def _overlay_heatmap(base_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = Image.fromarray(np.clip(base_image, 0, 255).astype('uint8')).convert('RGBA')
    normalized = _normalize_map(heatmap)
    colored = cm.get_cmap('jet')(normalized)[..., :3]
    colored_image = Image.fromarray((colored * 255).astype('uint8')).convert('RGBA')
    alpha_layer = Image.fromarray((normalized * alpha * 255).astype('uint8'))
    colored_image.putalpha(alpha_layer)
    return Image.alpha_composite(base, colored_image)


def _overlay_mask(base_image: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = Image.fromarray(np.clip(base_image, 0, 255).astype('uint8')).convert('RGBA')
    alpha_mask = Image.fromarray((mask.astype(np.uint8) * 170).astype('uint8'))
    tint = Image.new('RGBA', base.size, (34, 211, 238, 0))
    tint.putalpha(alpha_mask)
    return Image.alpha_composite(base, tint)


def _predict_from_display_batch(images: np.ndarray, classifier) -> np.ndarray:
    processed = np.stack([classifier.preprocess_array(image * 255.0 if image.max() <= 1.5 else image) for image in images], axis=0)
    return classifier.model.predict(processed, verbose=0)


def compute_gradcam(
    classifier,
    display_image: np.ndarray,
    class_index: int,
    last_conv_layer_name: str | None = None,
    threshold_quantile: float = 0.82,
) -> dict[str, Any]:
    model = classifier.model
    last_conv_layer_name = last_conv_layer_name or 'block5_conv3'
    if 'vgg16' not in [layer.name for layer in model.layers]:
        raise ValueError("The current Grad-CAM implementation expects a VGG16 backbone named 'vgg16'.")

    preprocessed = classifier.preprocess_array(display_image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(preprocessed, axis=0), dtype=tf.float32)
    conv_model, classifier_model = _build_vgg16_grad_models(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_output = conv_model(input_tensor)
        tape.watch(conv_output)
        predictions = classifier_model(conv_output)
        target_score = predictions[:, class_index]
    gradients = tape.gradient(target_score, conv_output)

    if gradients is None:
        raise ValueError('Gradients could not be computed for Grad-CAM.')

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_gradients, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()
    heatmap = gaussian(heatmap, sigma=1.1, preserve_range=True)
    heatmap = np.array(
        Image.fromarray((_normalize_map(heatmap) * 255).astype('uint8')).resize(
            (display_image.shape[1], display_image.shape[0]),
            Image.Resampling.BILINEAR,
        ),
        dtype='float32',
    ) / 255.0
    heatmap = _normalize_map(gaussian(heatmap, sigma=1.2, preserve_range=True))
    mask = _build_mask_from_heatmap(heatmap, quantile=threshold_quantile)

    peak_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    positive_values = heatmap[heatmap > 0]
    activation_intensity = float(np.percentile(positive_values, 95)) if positive_values.size else 0.0

    return {
        'input_tensor': input_tensor.numpy()[0],
        'heatmap': heatmap,
        'mask': mask,
        'peak': {
            'x': float(peak_index[1] / max(heatmap.shape[1] - 1, 1)),
            'y': float(peak_index[0] / max(heatmap.shape[0] - 1, 1)),
        },
        'activation_intensity': activation_intensity,
        'overlay': _overlay_heatmap(display_image, heatmap),
    }


def compute_lime(
    classifier,
    display_image: np.ndarray,
    class_index: int,
    num_samples: int = 700,
    n_segments: int = 160,
    num_features: int = 6,
) -> dict[str, Any]:
    if not LIME_AVAILABLE:
        return {
            'overlay': None,
            'mask': None,
            'notes': ['LIME not available in the current environment.'],
        }

    notes: list[str] = []
    display_float = np.clip(display_image / 255.0, 0.0, 1.0).astype('float32')
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        display_float,
        lambda images: _predict_from_display_batch(images, classifier),
        labels=(class_index,),
        top_labels=len(classifier.class_labels),
        num_samples=num_samples,
        hide_color=0.0,
        segmentation_fn=lambda image: slic(
            image,
            n_segments=n_segments,
            compactness=10,
            sigma=1.0,
            start_label=0,
            channel_axis=-1,
        ),
    )

    target_label = class_index if class_index in explanation.local_exp else explanation.top_labels[0]
    weights = explanation.local_exp.get(target_label, [])
    positive_segments = [segment_id for segment_id, weight in sorted(weights, key=lambda item: item[1], reverse=True) if weight > 0][:num_features]
    if not positive_segments:
        notes.append('LIME returned no stable positive superpixels for the requested class.')
        return {
            'overlay': None,
            'mask': np.zeros(display_image.shape[:2], dtype=np.uint8),
            'notes': notes,
        }

    lime_mask = np.isin(explanation.segments, positive_segments).astype(np.uint8)
    lime_mask = _clean_mask(lime_mask, min_pixels=18)

    base = Image.fromarray(np.clip(display_image, 0, 255).astype('uint8')).convert('RGBA')
    boundaries = find_boundaries(lime_mask, mode='outer')
    tint = Image.new('RGBA', base.size, (251, 191, 36, 0))
    tint.putalpha(Image.fromarray((boundaries.astype(np.uint8) * 255).astype('uint8')))
    overlay = Image.alpha_composite(base, tint)

    return {
        'overlay': overlay,
        'mask': lime_mask,
        'notes': notes,
    }


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray | None) -> float | None:
    if mask_b is None:
        return None
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def _align_masks(gradcam_mask: np.ndarray, lime_mask: np.ndarray | None, iou: float | None) -> np.ndarray:
    if lime_mask is None:
        return gradcam_mask.astype(np.uint8)
    if iou is not None and iou >= 0.15:
        consensus = np.logical_and(gradcam_mask > 0, lime_mask > 0)
        if np.count_nonzero(consensus) > 12:
            return _clean_mask(consensus, min_pixels=12)
    guided_union = np.logical_or(gradcam_mask > 0, binary_closing(lime_mask > 0, footprint=disk(1)))
    return _clean_mask(guided_union, min_pixels=18)


class ExplainabilityGenerator:
    def generate(self, image_path: str, classifier, class_index: int) -> ExplainabilityResult:
        display_image = classifier.load_display_array(image_path)
        gradcam_result = compute_gradcam(classifier, display_image, class_index)
        lime_result = compute_lime(classifier, display_image, class_index)
        overlap_iou = compute_iou(gradcam_result['mask'], lime_result['mask'])
        aligned_mask = _align_masks(gradcam_result['mask'], lime_result['mask'], overlap_iou)

        area_px, area_ratio, spread_score, masked_intensity = _compute_mask_metrics(aligned_mask, gradcam_result['heatmap'])
        estimated_volume_cm3 = _estimate_volume_cm3(area_px)
        mask_overlay = _overlay_mask(display_image, aligned_mask)
        notes = list(lime_result['notes'])
        if overlap_iou is not None and overlap_iou < 0.10:
            notes.append('Grad-CAM and LIME disagree strongly on lesion localization; review the slice and model confidence carefully.')
        elif overlap_iou is not None and overlap_iou >= 0.35:
            notes.append('Grad-CAM and LIME show moderate spatial agreement on the highlighted lesion region.')

        return ExplainabilityResult(
            uploaded_image=_to_data_url(Image.fromarray(np.clip(display_image, 0, 255).astype('uint8'))),
            gradcam_overlay=_to_data_url(gradcam_result['overlay']),
            tumor_mask_overlay=_to_data_url(mask_overlay),
            lime_overlay=_to_data_url(lime_result['overlay']) if lime_result['overlay'] is not None else None,
            gradcam_peak=gradcam_result['peak'],
            lime_available=lime_result['overlay'] is not None,
            notes=notes,
            tumor_area_px=area_px,
            tumor_area_ratio=area_ratio,
            estimated_volume_cm3=estimated_volume_cm3,
            activation_intensity=float(max(gradcam_result['activation_intensity'], masked_intensity)),
            spread_score=spread_score,
            overlap_iou=overlap_iou,
            pixel_spacing_mm=PIXEL_SPACING_MM,
            slice_thickness_mm=SLICE_THICKNESS_MM,
        )
