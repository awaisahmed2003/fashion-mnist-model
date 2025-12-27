import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Optional: make TF a bit more deterministic (may slow a bit)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

train_df = pd.read_csv(r"C:\Users\helpi\OneDrive\Desktop\DEEP LEARNING PROJECT DATASETS\train.csv")
test_df  = pd.read_csv(r"C:\Users\helpi\OneDrive\Desktop\DEEP LEARNING PROJECT DATASETS\test.csv")


if "label" not in train_df.columns:
    raise ValueError("train.csv must contain a 'label' column.")git add .

    raise ValueError("test.csv must contain an 'id' column.")


pixel_cols = [c for c in train_df.columns if c != "label"]
if len(pixel_cols) != 784:

    raise ValueError(f"Expected 784 pixel columns, found {len(pixel_cols)}. Columns: {train_df.columns.tolist()[:20]} ...")

test_pixel_cols = [c for c in test_df.columns if c != "id"]
if len(test_pixel_cols) != 784:
    raise ValueError(f"Expected 784 pixel columns in test.csv, found {len(test_pixel_cols)}.")


test_df = test_df[["id"] + pixel_cols] if set(pixel_cols) == set(test_pixel_cols) else test_df


# 3) Prepare arrays

X = train_df[pixel_cols].to_numpy(dtype=np.float32) / 255.0
y = train_df["label"].to_numpy(dtype=np.int64)

test_ids = test_df["id"].to_numpy()
test_X = test_df[pixel_cols].to_numpy(dtype=np.float32) / 255.0

# Reshape to (N, 28, 28, 1)
X = X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)


# 4) Train/Validation split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.15,
    random_state=SEED,
    stratify=y
)



# 5) Build model

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),


        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),


        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model()
model.summary()


# 6) Train with callbacks

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.3, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy: {val_acc*100:.2f}%")


# 7) Train final model on ALL data

final_model = build_model()


best_epochs = len(history.history["loss"])
print(f"\nRetraining on full data for {best_epochs} epochs...")

final_model.fit(
    X, y,
    epochs=best_epochs,
    batch_size=64,
    verbose=1
)

# 8) Predict on test set

probs = final_model.predict(test_X, batch_size=256, verbose=1)
pred_labels = np.argmax(probs, axis=1).astype(int)

tta_aug = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.10),
])

def tta_predict(model, X, n=5):
    probs = []
    for _ in range(n):
        X_aug = tta_aug(X, training=True)
        probs.append(model.predict(X_aug, verbose=0))
    return np.mean(probs, axis=0)

#probs = tta_predict(final_model, test_X, n=5)
#pred_labels = np.argmax(probs, axis=1)



# 9) Create submission.csv

submission = pd.DataFrame({
    "id": test_ids,
    "label": pred_labels
})
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv with shape:", submission.shape)
print(submission.head())
