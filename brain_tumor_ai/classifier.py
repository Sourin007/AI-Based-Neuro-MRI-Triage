from __future__ import annotations

from functools import lru_cache
import json
import os

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.models import load_model

from .config import Settings
from .state import PredictionResult


DEFAULT_CLASS_LABELS = ["pituitary", "glioma", "notumor", "meningioma"]
DEFAULT_IMAGE_SIZE = 128


@lru_cache(maxsize=1)
def _load_classifier(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path, compile=False)


@lru_cache(maxsize=1)
def _load_metadata(model_path: str) -> dict:
    metadata_path = os.path.splitext(model_path)[0] + '.metadata.json'
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


class BrainTumorClassifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._metadata = _load_metadata(settings.model_path)

    @property
    def model(self):
        return _load_classifier(self.settings.model_path)

    @property
    def class_labels(self) -> list[str]:
        labels = self._metadata.get('class_labels')
        if labels:
            return list(labels)
        return DEFAULT_CLASS_LABELS

    @property
    def image_size(self) -> int:
        metadata_size = self._metadata.get('image_size')
        if metadata_size:
            return int(metadata_size)
        input_shape = getattr(self.model, 'input_shape', None)
        if isinstance(input_shape, tuple) and len(input_shape) >= 3 and input_shape[1]:
            return int(input_shape[1])
        return DEFAULT_IMAGE_SIZE

    @property
    def preprocess_mode(self) -> str:
        return str(self._metadata.get('preprocess_mode', 'scale_0_1'))

    def load_display_array(self, image_path: str) -> np.ndarray:
        image = load_img(image_path, target_size=(self.image_size, self.image_size), color_mode='rgb')
        image_array = img_to_array(image).astype('float32')
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        if image_array.shape[-1] == 1:
            image_array = np.repeat(image_array, 3, axis=-1)
        return image_array

    def preprocess_array(self, image_array: np.ndarray) -> np.ndarray:
        processed = image_array.astype('float32')
        if processed.ndim == 2:
            processed = np.stack([processed] * 3, axis=-1)
        if processed.shape[-1] == 1:
            processed = np.repeat(processed, 3, axis=-1)

        if self.preprocess_mode == 'vgg16':
            return vgg16_preprocess_input(processed.copy())
        return processed / 255.0

    def load_image_array(self, image_path: str) -> np.ndarray:
        return self.preprocess_array(self.load_display_array(image_path))

    def predict_probabilities(self, image_path: str) -> dict[str, float]:
        model = self.model
        image_array = self.load_image_array(image_path)
        batch = np.expand_dims(image_array, axis=0)
        probabilities = model.predict(batch, verbose=0)[0].astype('float64')
        total = float(np.sum(probabilities))
        if total <= 0:
            probabilities = np.full_like(probabilities, 1 / len(probabilities), dtype='float64')
        else:
            probabilities = probabilities / total
        return {label: float(score) for label, score in zip(self.class_labels, probabilities.tolist())}

    def predict(self, image_path: str) -> PredictionResult:
        class_probabilities = self.predict_probabilities(image_path)
        sorted_probs = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)
        raw_label, raw_confidence = sorted_probs[0]
        _, second_confidence = sorted_probs[1]
        margin = raw_confidence - second_confidence

        adjusted_label = raw_label
        correction_reason = None
        review_recommended = raw_confidence < self.settings.confidence_threshold or margin < self.settings.review_margin_threshold

        meningioma_prob = class_probabilities.get('meningioma', 0.0)
        suspicious_no_tumor = (
            raw_label == 'notumor'
            and meningioma_prob >= self.settings.meningioma_override_probability
            and (
                raw_confidence < self.settings.no_tumor_acceptance_threshold
                or meningioma_prob >= raw_confidence * self.settings.meningioma_override_ratio
            )
        )
        if suspicious_no_tumor:
            adjusted_label = 'meningioma'
            correction_reason = (
                'The raw prediction favored no_tumor, but the meningioma probability was materially elevated. '
                'The case has been escalated as a meningioma-suspected tumor for clinician review.'
            )
            review_recommended = True

        binary_label = 'no_tumor' if adjusted_label == 'notumor' else 'tumor'
        decision_status = 'review_required' if review_recommended else 'accepted'
        adjusted_confidence = class_probabilities[adjusted_label]

        return PredictionResult(
            binary_label=binary_label,
            subtype_label=adjusted_label,
            confidence=float(adjusted_confidence),
            class_probabilities=class_probabilities,
            raw_subtype_label=raw_label,
            raw_confidence=float(raw_confidence),
            review_recommended=review_recommended,
            decision_status=decision_status,
            correction_reason=correction_reason,
        )
