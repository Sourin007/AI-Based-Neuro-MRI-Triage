from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from brain_tumor_ai.classifier import BrainTumorClassifier
from brain_tumor_ai.config import Settings

CANONICAL_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
TEST_FOLDER_MAP = {
    "giloma": "glioma",
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "notumor",
    "pituitary": "pituitary",
}


def evaluate(test_root: Path, model_path: str | None = None, show_plot: bool = True):
    settings = Settings.from_env()
    if model_path:
        settings = replace(settings, model_path=model_path)
    classifier = BrainTumorClassifier(settings)

    true_labels: list[str] = []
    raw_preds: list[str] = []
    adjusted_preds: list[str] = []

    for folder in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        true_label = TEST_FOLDER_MAP[folder.name.lower()]
        for file in sorted([f for f in folder.iterdir() if f.is_file()]):
            prediction = classifier.predict(str(file))
            true_labels.append(true_label)
            raw_preds.append(prediction.raw_subtype_label)
            adjusted_preds.append(prediction.subtype_label)

    raw_accuracy = accuracy_score(true_labels, raw_preds)
    adjusted_accuracy = accuracy_score(true_labels, adjusted_preds)

    print(f"Evaluated model: {settings.model_path}")
    print(f"Testing samples: {len(true_labels)}")
    print(f"Raw testing accuracy: {raw_accuracy:.4%}")
    print(f"Adjusted testing accuracy: {adjusted_accuracy:.4%}")
    print()

    print("Raw model classification report")
    print(classification_report(true_labels, raw_preds, labels=CANONICAL_LABELS, digits=4))
    print("Adjusted inference classification report")
    print(classification_report(true_labels, adjusted_preds, labels=CANONICAL_LABELS, digits=4))

    raw_cm = confusion_matrix(true_labels, raw_preds, labels=CANONICAL_LABELS)
    adjusted_cm = confusion_matrix(true_labels, adjusted_preds, labels=CANONICAL_LABELS)
    print("Raw confusion matrix")
    print(raw_cm)
    print("Adjusted confusion matrix")
    print(adjusted_cm)

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(raw_cm, annot=True, fmt='d', cmap='Blues', xticklabels=CANONICAL_LABELS, yticklabels=CANONICAL_LABELS, ax=axes[0])
        axes[0].set_title('Raw Predictions')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        sns.heatmap(adjusted_cm, annot=True, fmt='d', cmap='Greens', xticklabels=CANONICAL_LABELS, yticklabels=CANONICAL_LABELS, ax=axes[1])
        axes[1].set_title('Adjusted Predictions')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the brain tumor classifier on the testing split.')
    parser.add_argument('--test-root', default=r'D:\year project\MRI IMAGES\Testing')
    parser.add_argument('--model-path', default=None, help='Optional path to a specific model file to evaluate.')
    parser.add_argument('--no-plot', action='store_true', help='Disable confusion matrix plots.')
    args = parser.parse_args()
    evaluate(Path(args.test_root), model_path=args.model_path, show_plot=not args.no_plot)
