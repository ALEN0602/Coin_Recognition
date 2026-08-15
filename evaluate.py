import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

MODELS = {
    "Baseline": "models/coin_classifier.keras",
    "FineTuned": "models/coin_classifier_finetuned.keras"
}
HISTORY_PATHS = {
    "Baseline": "models/coin_classifier_history.pkl",
    "FineTuned": "models/coin_classifier_finetuned_history.pkl",
}

TEST_DIR = "dataset/test"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())

def load_history(path):
    if not path or not os.path.exists(path):
        return None

    try:
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [!] Could not load history from {path}: {e}")
        return None


def plot_training_curves(history, model_name):
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"

    has_acc = acc_key in history and val_acc_key in history
    has_loss = "loss" in history and "val_loss" in history

    if not has_acc and not has_loss:
        print(f"  [!] History file for {model_name} has no recognizable "
              f"accuracy/loss keys - skipping curves. Keys found: "
              f"{list(history.keys())}")
        return None

    epochs = range(1, len(history.get("loss", history.get(acc_key, []))) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if has_acc:
        axes[0].plot(epochs, history[acc_key], label="Training Accuracy", marker="o")
        axes[0].plot(epochs, history[val_acc_key], label="Validation Accuracy", marker="o")
        axes[0].set_title(f"{model_name}: Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].axis("off")
        axes[0].set_title("Accuracy data not available")

    if has_loss:
        axes[1].plot(epochs, history["loss"], label="Training Loss", marker="o")
        axes[1].plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
        axes[1].set_title(f"{model_name}: Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].set_title("Loss data not available")

    plt.suptitle(f"{model_name} Training History")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 60)
    print(f"Evaluating: {model_name}")
    print("=" * 60)

    model = load_model(model_path)

    history_path = HISTORY_PATHS.get(model_name)
    history = load_history(history_path)

    if history is not None:
        curves_path = plot_training_curves(history, model_name)
        if curves_path:
            print(f"\nSaved training curves: {curves_path}")
    else:
        print(f"\n[!] No training history found at "
              f"'{history_path}' - skipping accuracy/loss curves for "
              f"{model_name}. Save history.history during training to "
              f"enable this (see module docstring at the top of this file).")

    test_loss, test_accuracy = model.evaluate(
        test_gen,
        verbose=1
    )

    predictions = model.predict(
        test_gen,
        verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes


    print("\nTest Loss     :", round(test_loss, 4))
    print("Test Accuracy :", f"{test_accuracy*100:.2f}%")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print("\nClassification Report")
    print(report)

    report_file = os.path.join(
        RESULTS_DIR,
        f"{model_name}_classification_report.txt"
    )

    with open(report_file, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    np.savetxt(
        os.path.join(
            RESULTS_DIR,
            f"{model_name}_confusion_matrix.txt"
        ),
        cm,
        fmt="%d"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{model_name}_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close(fig)

    print("\nSaved:")
    print(report_file)
    print(os.path.join(
        RESULTS_DIR,
        f"{model_name}_confusion_matrix.png"
    ))

print("\n============================================")
print("Evaluation of all models completed.")
print("============================================")