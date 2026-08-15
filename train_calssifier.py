import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
DATA_DIR = "dataset"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=180,
    brightness_range=[0.7, 1.3],
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.30),
    layers.Dense(64, activation="relu"),
    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)


history_path = os.path.join(MODEL_DIR, "coin_classifier_history.pkl")
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print(f"\nTraining history saved to: {history_path}")

print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(test_gen)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy*100:.2f}%")


model_path = os.path.join(MODEL_DIR, "coin_classifier.keras")
model.save(model_path)

print(f"\nModel saved to: {model_path}")

print("\nClass Mapping:")
print(train_gen.class_indices)

print("\nTraining completed successfully.")