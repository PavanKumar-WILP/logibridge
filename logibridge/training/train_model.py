#!/usr/bin/env python3

"""
---------------------------------------------------------
LogiEdge
Component D

Train MLP Classifier

Produces

models/model.keras

models/model.tflite (later)

---------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import random

random.seed(42)

np.random.seed(42)

tf.random.set_seed(42)

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "training" / "dataset.csv"

MODEL_DIR = ROOT / "training" / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.keras"

STATS_FILE = ROOT / "data_pipeline" / "training_stats.npy"

# ------------------------------------------------------
# Load Dataset
# ------------------------------------------------------

df = pd.read_csv(DATASET)
if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}\n"
        "Run generate_dataset.py first."
    )

print(df.head())

print("\nDataset Shape")

print(df.shape)

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values


# ------------------------------------------------------
# Train Test Split
# ------------------------------------------------------
print("\nClass Distribution")
print(df["label"].value_counts())
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print()

print("Training Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# ------------------------------------------------------
# Load Normalization Statistics
# ------------------------------------------------------

stats = np.load(

    STATS_FILE,

    allow_pickle=True

).item()

mean = stats["mean"]

std = stats["std"]

X_train = (X_train - mean) / std
if X_train.shape[1] != len(mean):
    raise ValueError(
        "Feature dimension mismatch "
        "between dataset and training statistics."
    )

X_test = (X_test - mean) / std

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# ------------------------------------------------------
# Build MLP
# ------------------------------------------------------

model = Sequential(

    [

        Dense(

            32,

            activation="relu",

            input_shape=(6,)

        ),

        Dense(

            16,

            activation="relu"

        ),

        Dense(

            3,

            activation="softmax"

        )

    ]

)

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# ------------------------------------------------------
# Train
# ------------------------------------------------------

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)
# ------------------------------------------------------
# Evaluation
# ------------------------------------------------------

prediction = model.predict(
    X_test,
    verbose=0
)

prediction = np.argmax(
    prediction,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Test Accuracy : {accuracy:.4f}")

report = classification_report(
    y_test,
    prediction,
    target_names=[
        "Normal",
        "Warning",
        "Critical"
    ],
    output_dict=True
)

critical_recall = report["Critical"]["recall"]

print(f"Critical Recall     : {critical_recall:.4f}")

print()

print(classification_report(
    y_test,
    prediction,
    target_names=[
        "Normal",
        "Warning",
        "Critical"
    ]
))

print()

print("Confusion Matrix")

print(confusion_matrix(
    y_test,
    prediction
))

print()

if accuracy >= 0.88:
    print("Accuracy requirement satisfied")
else:
    print("Accuracy below required threshold")

if critical_recall >= 0.95:
    print("Critical Recall requirement satisfied")
else:
    print("Critical Recall below required threshold")

# ------------------------------------------------------
# Mandatory +3 Sigma Experiment
# ------------------------------------------------------

shifted_mean = mean + (3 * std)

X_shifted = (X_test - shifted_mean) / std

prediction = model.predict(

    X_shifted,

    verbose=0

)

prediction = np.argmax(

    prediction,

    axis=1

)

shift_accuracy = accuracy_score(

    y_test,

    prediction

)

print()

print("=" * 50)

print("Accuracy with shifted statistics")

print("=" * 50)

print(shift_accuracy)

# ------------------------------------------------------
# Save Model
# ------------------------------------------------------

model.save(

    MODEL_PATH

)
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()

print("=" * 60)

print("FINAL MODEL")

print("=" * 60)

print(f"Loss     : {loss:.4f}")

print(f"Accuracy : {accuracy:.4f}")
print()

print("Model Saved")

print(MODEL_PATH)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.grid(True)

plt.savefig(
    MODEL_DIR / "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend([
    "Train",
    "Validation"
])

plt.grid(True)

plt.savefig(
    MODEL_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training curves saved.")

