#!/usr/bin/env python3

"""
---------------------------------------------------------
LogiEdge

Component F1

Model Variant M3

35% Pruning
+
INT8 Quantization

Produces

training/models/pruned_model.keras

inference/pruned_model.tflite
---------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tf_keras
from tf_keras.models import Sequential
from tf_keras.layers import Input, Dense

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "training" / "dataset.csv"

MODEL = ROOT / "training" / "models" / "model.keras"

PRUNED_MODEL = ROOT / "training" / "models" / "pruned_model.keras"

TFLITE_MODEL = ROOT / "inference" / "pruned_model.tflite"

STATS = ROOT / "data_pipeline" / "training_stats.npy"

##########################################################
# Load dataset
##########################################################

df = pd.read_csv(DATASET)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

stats = np.load(STATS, allow_pickle=True).item()

mean = stats["mean"]
std = stats["std"]


X = (X - mean) / std
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

##########################################################
# Load trained model
##########################################################

model = Sequential([
    Input(shape=(6,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
])

weights_data = np.load(
    ROOT / "training" / "models" / "model_weights.npz"
)

weights = [
    weights_data[name]
    for name in weights_data.files
]

model.set_weights(weights)
print("Original model weights:", len(model.get_weights()))

##########################################################
# Apply pruning
##########################################################

batch_size = 32
epochs = 15

end_step = np.ceil(len(X_train) / batch_size).astype(np.int32) * epochs

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.35,
    begin_step=0,
    end_step=end_step
)

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule=pruning_schedule
)

pruned_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

##########################################################
# Fine tune
##########################################################

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

pruned_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

##########################################################
# Strip pruning wrappers
##########################################################

final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

##########################################################
# Evaluate
##########################################################

pred = final_model.predict(X_test, verbose=0)

pred = np.argmax(pred, axis=1)

acc = accuracy_score(y_test, pred)

print("\nPruned Accuracy =", acc)

##########################################################
# Save pruned model
##########################################################

final_model.save(PRUNED_MODEL)

print("\nSaved")

print(PRUNED_MODEL)

##########################################################
# Representative dataset
##########################################################

def representative_dataset():

    for i in range(min(200, len(X_train))):

        yield [X_train[i:i+1]]

##########################################################
# Convert INT8
##########################################################

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8

converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_MODEL, "wb") as f:

    f.write(tflite_model)

print("\nSaved")

print(TFLITE_MODEL)

print("\nSize")

print(round(TFLITE_MODEL.stat().st_size / 1024, 2), "KB")