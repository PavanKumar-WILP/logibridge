#!/usr/bin/env python3

"""
----------------------------------------------------------
Component D3

Structured Pruning
↓

Fine Tune

↓

INT8 Quantisation

----------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "training" / "dataset.csv"

MODEL_DIR = ROOT / "training" / "models"

MODEL = MODEL_DIR / "model.keras"

PRUNED_MODEL = MODEL_DIR / "model_pruned.keras"

PRUNED_TFLITE = MODEL_DIR / "model_pruned.tflite"

STATS = ROOT / "data_pipeline" / "training_stats.npy"

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = pd.read_csv(DATASET)

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values

stats = np.load(

    STATS,

    allow_pickle=True

).item()

mean = stats["mean"]

std = stats["std"]

X = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ----------------------------------------------------
# Load FP32 Model
# ----------------------------------------------------

model = tf.keras.models.load_model(

    MODEL

)

print(model.summary())

# ----------------------------------------------------
# Apply Pruning
# ----------------------------------------------------

pruning_params = {

    "pruning_schedule":

    tfmot.sparsity.keras.PolynomialDecay(

        initial_sparsity=0.20,

        final_sparsity=0.50,

        begin_step=0,

        end_step=1000

    )

}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(

    model,

    **pruning_params

)

# ----------------------------------------------------
# Compile
# ----------------------------------------------------

pruned_model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ----------------------------------------------------
# Fine Tune
# ----------------------------------------------------

callbacks = [

    tfmot.sparsity.keras.UpdatePruningStep()

]

pruned_model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=10,

    batch_size=32,

    callbacks=callbacks,

    verbose=1

)

# ----------------------------------------------------
# Remove Pruning Wrappers
# ----------------------------------------------------

final_model = tfmot.sparsity.keras.strip_pruning(

    pruned_model

)

final_model.save(

    PRUNED_MODEL

)

# ----------------------------------------------------
# Convert to TFLite
# ----------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(

    final_model

)

converter.optimizations = [

    tf.lite.Optimize.DEFAULT

]

tflite_model = converter.convert()

with open(

    PRUNED_TFLITE,

    "wb"

) as f:

    f.write(tflite_model)

print()

print("Pruned model saved")

print(PRUNED_MODEL)

print()

print("Pruned TFLite saved")

print(PRUNED_TFLITE)

import os

fp32 = os.path.getsize(MODEL)

pruned = os.path.getsize(PRUNED_MODEL)

tflite = os.path.getsize(PRUNED_TFLITE)

print()

print("=" * 60)

print("MODEL SIZE COMPARISON")

print("=" * 60)

print(f"FP32           : {fp32/1024:.2f} KB")

print(f"Pruned         : {pruned/1024:.2f} KB")

print(f"Pruned TFLite  : {tflite/1024:.2f} KB")


