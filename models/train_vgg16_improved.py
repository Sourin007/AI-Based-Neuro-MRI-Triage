from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

CANONICAL_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CANONICAL_CLASS_NAMES)}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def focal_loss(gamma: float = 2.0, alpha: float = 0.35):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weights = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weights * cross_entropy, axis=-1)

    return loss_fn


def normalize_class_name(name: str) -> str:
    return {"giloma": "glioma"}.get(name.lower(), name.lower())


def collect_samples(directory: Path) -> tuple[list[str], list[int]]:
    file_paths: list[str] = []
    labels: list[int] = []
    for class_dir in sorted([path for path in directory.iterdir() if path.is_dir()]):
        normalized_name = normalize_class_name(class_dir.name)
        if normalized_name not in CLASS_TO_INDEX:
            continue
        for file in sorted([path for path in class_dir.iterdir() if path.is_file()]):
            file_paths.append(str(file))
            labels.append(CLASS_TO_INDEX[normalized_name])
    return file_paths, labels


def decode_image(path: tf.Tensor, label: tf.Tensor):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return image, label


def build_dataset(directory: Path, training: bool) -> tf.data.Dataset:
    file_paths, labels = collect_samples(directory)
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        dataset = dataset.shuffle(len(file_paths), reshuffle_each_iteration=True)

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    def preprocess(path: tf.Tensor, label: tf.Tensor):
        image, label = decode_image(path, label)
        if training:
            image = augmentation(image, training=True)
        image = preprocess_input(image)
        return image, label

    return dataset.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_model(num_classes: int):
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2.0, alpha=0.35),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an improved VGG16 brain tumor classifier.")
    parser.add_argument("--data-root", default=r"D:\year project\MRI IMAGES", help="Dataset root containing Training/ and Testing/ folders")
    parser.add_argument("--output-model", default=r"D:\year project\models\model_vgg16_improved.keras", help="Path to save the trained model")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"

    train_files, train_labels = collect_samples(train_dir)
    test_files, _ = collect_samples(test_dir)
    print(f"Found {len(train_files)} training files across {len(CANONICAL_CLASS_NAMES)} classes.")
    print(f"Found {len(test_files)} testing files across {len(CANONICAL_CLASS_NAMES)} normalized classes.")

    train_dataset = build_dataset(train_dir, training=True)
    test_dataset = build_dataset(test_dir, training=False)

    y_train = np.array(train_labels, dtype=np.int32)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CANONICAL_CLASS_NAMES)), y=y_train)
    class_weight_map = {index: float(weight) for index, weight in enumerate(class_weights)}

    model = build_model(num_classes=len(CANONICAL_CLASS_NAMES))
    callbacks = [
        ModelCheckpoint(args.output_model, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=30,
        class_weight=class_weight_map,
        callbacks=callbacks,
        verbose=1,
    )

    metadata = {
        "class_labels": CANONICAL_CLASS_NAMES,
        "image_size": IMAGE_SIZE[0],
        "preprocess_mode": "vgg16",
        "class_weight": class_weight_map,
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
    }
    metadata_path = Path(args.output_model).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output_model}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
