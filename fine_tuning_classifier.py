import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

MODEL_PATH = "models/coin_classifier.keras"
OUTPUT_PATH = "models/coin_classifier_finetuned.keras"
HISTORY_PATH = "models/coin_classifier_finetuned_history.pkl"

train_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nBaseline model loaded.\n")

model.summary()

base_model = None

for layer in model.layers:
    if "mobilenet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise RuntimeError("Could not locate MobileNetV2 backbone.")

print("\nFound backbone:", base_model.name)

base_model.trainable = True

for layer in base_model.layers[:-15]:
    layer.trainable = False

print("\nTrainable Layers:")

trainable = sum(layer.trainable for layer in base_model.layers)

print(
    f"\nTrainable MobileNetV2 layers: "
    f"{trainable}/{len(base_model.layers)}"
)

model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        OUTPUT_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks
)

print("\nFine-tuning completed.")

os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved to: {HISTORY_PATH}")

model.save(OUTPUT_PATH)

print(f"\nSaved to {OUTPUT_PATH}")