#!/usr/bin/env python3

"""
----------------------------------------------------------

Component D2

True INT8 Post Training Quantization

Input

    model.keras

Output

    inference/model.tflite

----------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "training" / "models" / "model.keras"

DATASET = ROOT / "training" / "dataset.csv"

OUTPUT = ROOT / "inference" / "model.tflite"
print("Running:", __file__)
print("Dataset :", DATASET)
OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading Keras model...")

model = tf.keras.models.load_model(MODEL)

# ----------------------------------------------------------
# Representative Dataset
# ----------------------------------------------------------



df = pd.read_csv(DATASET)

X = df.drop(columns=["label"]).to_numpy(dtype="float32")
y = df["label"].to_numpy()

# ----------------------------------------------------------
# Normalize representative dataset exactly like training
# ----------------------------------------------------------

STATS_FILE = ROOT / "data_pipeline" / "training_stats.npy"

stats = np.load(
    STATS_FILE,
    allow_pickle=True
).item()

mean = stats["mean"].astype(np.float32)
std = stats["std"].astype(np.float32)

std[std == 0] = 1.0

X = (X - mean) / std

def representative_dataset():

    print(f"Representative samples: {len(X)}")

    for sample in X:

        sample = sample.reshape(1, 6).astype(np.float32)

        yield [sample]
        

# ----------------------------------------------------------
# Configure Converter
# ----------------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8

converter.inference_output_type = tf.int8

# ----------------------------------------------------------
# Convert
# ----------------------------------------------------------

print()

print("Converting model...")

tflite_model = converter.convert()

with open(
    OUTPUT,
    "wb"
) as f:

    f.write(tflite_model)

print()

print("INT8 Model Saved")

print(OUTPUT)

# ----------------------------------------------------------
# Compare Sizes
# ----------------------------------------------------------

keras_size = MODEL.stat().st_size

tflite_size = OUTPUT.stat().st_size

print()

print("=" * 60)

print("MODEL SIZE")

print("=" * 60)

print(f"Keras Model : {keras_size/1024:.2f} KB")

print(f"INT8 Model  : {tflite_size/1024:.2f} KB")

reduction = (

    100

    * (keras_size - tflite_size)

    / keras_size

)

print()

print(f"Reduction : {reduction:.2f}%")

# ----------------------------------------------------------
# Verify Model
# ----------------------------------------------------------

print()

print("Verifying model...")

interpreter = tf.lite.Interpreter(
    model_path=str(OUTPUT)
)

interpreter.allocate_tensors()

print()

print("Verification successful.")

print()

print("Input")

print(interpreter.get_input_details())

print()

print("Output")

print(interpreter.get_output_details())

     