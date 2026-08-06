#!/usr/bin/env python3

"""
---------------------------------------------------------

LogiBridge

Component F2

Benchmark Three Model Variants

Benchmarks

M1 - FP32 Baseline
M2 - PTQ INT8
M3 - Structured Pruning + PTQ INT8

Measures

1. Mean inference latency (200 runs)
2. P95 inference latency
3. Model size
4. Classification accuracy
5. Energy per inference

---------------------------------------------------------
"""

import os
import time
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

##########################################################
# Paths
##########################################################

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "training" / "dataset.csv"

TRAINING_STATS = ROOT / "data_pipeline" / "training_stats.npy"

RESULTS_DIR = ROOT / "optimisation" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

##########################################################
# Model Paths
##########################################################

M1_WEIGHTS = ROOT / "training" / "models" / "model.weights.h5"

M2_MODEL = ROOT / "inference" / "model.tflite"

M3_MODEL = ROOT / "inference" / "pruned_model.tflite"

##########################################################
# Benchmark Parameters
##########################################################

WARMUP_RUNS = 10

BENCHMARK_RUNS = 200

CPU_TDP_WATTS = 15.0

##########################################################
# Load Dataset
##########################################################

if not DATASET.exists():

    raise FileNotFoundError(DATASET)

df = pd.read_csv(DATASET)

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values

stats = np.load(
    TRAINING_STATS,
    allow_pickle=True
).item()

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


print("=" * 65)
print("LogiBridge Benchmark")
print("=" * 65)
print()

print(f"Validation Samples : {len(X_test)}")
print()

##########################################################
# Helper Function
##########################################################

def calculate_metrics(

    latencies,

    accuracy,

    model_size_kb,

    cpu_percent,

    memory_mb,

    variant

):

    latency_mean = float(np.mean(latencies))

    latency_p95 = float(np.percentile(latencies, 95))

    #######################################################
    # Energy Estimation
    # E = P × t
    #######################################################

    effective_power = CPU_TDP_WATTS * (cpu_percent / 100.0)

    energy = effective_power * (latency_mean / 1000.0)

    energy *= 1000          # Joules -> milliJoules

    return {

        "Variant": variant,

        "Mean Latency (ms)": round(latency_mean, 4),

        "P95 Latency (ms)": round(latency_p95, 4),

        "Model Size (KB)": round(model_size_kb, 2),

        "Accuracy (%)": round(accuracy * 100, 2),

        "Energy (mJ)": round(energy, 4),

        "CPU (%)": round(cpu_percent, 2),

        "Memory (MB)": round(memory_mb, 2)

    }

##########################################################
# Benchmark Keras FP32 Model (M1)
##########################################################

def benchmark_keras(weights_path, variant):

    print(f"Benchmarking {variant}")

    model = tf.keras.models.load_model(
        ROOT / "training/models/model_fp32.h5",
        compile=False
    )

    ######################################################
    # Dummy input
    ######################################################

    dummy = np.random.rand(
        1,
        6
    ).astype(np.float32)

    ######################################################
    # Warm-up (excluded from timing)
    ######################################################

    for _ in range(WARMUP_RUNS):

        _ = model.predict(
            dummy,
            verbose=0
        )

    ######################################################
    # Benchmark
    ######################################################

    process = psutil.Process()

    psutil.cpu_percent(interval=None)

    cpu_before = psutil.cpu_percent(interval=0.25)

    memory_after = process.memory_info().rss

    ######################################################
    # Actual Benchmark (200 timed runs)
    ######################################################

    latencies = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        _ = model.predict(
            dummy,
            verbose=0
        )

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    cpu_after = psutil.cpu_percent(interval=0.25)

    cpu_percent = (cpu_before + cpu_after) / 2

    ######################################################
    # Accuracy on validation set
    ######################################################

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    ######################################################
    # Model size
    ######################################################

    model_size = os.path.getsize(
        weights_path
    ) / 1024

    ######################################################
    # Memory used
    ######################################################

    memory_mb = memory_after / (1024 * 1024)

    ######################################################
    # Return metrics
    ######################################################

    return calculate_metrics(

        latencies=latencies,

        accuracy=accuracy,

        model_size_kb=model_size,

        cpu_percent=cpu_percent,

        memory_mb=memory_mb,

        variant=variant

    )

##########################################################
# Benchmark TensorFlow Lite Models (M2 / M3)
##########################################################

def benchmark_tflite(model_path, variant):

    print(f"Benchmarking {variant}")

    ######################################################
    # Load Interpreter
    ######################################################

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    ######################################################
    # Input Information
    ######################################################

    input_dtype = input_details[0]["dtype"]

    input_index = input_details[0]["index"]

    output_index = output_details[0]["index"]

    scale, zero_point = input_details[0]["quantization"]

    ######################################################
    # Dummy Input
    ######################################################

    dummy = np.random.rand(
        1,
        6
    ).astype(np.float32)

    if input_dtype == np.int8:

        dummy = np.round(
            dummy / scale + zero_point
        ).astype(np.int8)

    ######################################################
    # Warm-up (excluded)
    ######################################################

    for _ in range(WARMUP_RUNS):

        interpreter.set_tensor(
            input_index,
            dummy
        )

        interpreter.invoke()

        interpreter.get_tensor(
            output_index
        )

    ######################################################
    # Benchmark
    ######################################################


    process = psutil.Process()

    psutil.cpu_percent(interval=None)

    cpu_before = psutil.cpu_percent(interval=0.25)

    latencies = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        interpreter.set_tensor(
            input_index,
            dummy
        )

        interpreter.invoke()

        interpreter.get_tensor(
            output_index
        )

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    cpu_after = psutil.cpu_percent(interval=0.25)

    cpu_percent = (cpu_before + cpu_after) / 2

    memory_after = process.memory_info().rss

    

    ######################################################
    # Accuracy
    ######################################################

    predictions = []

    for sample in X_test:

        sample = sample.reshape(1, 6).astype(np.float32)

        if input_dtype == np.int8:

            sample = np.round(
                sample / scale + zero_point
            ).astype(np.int8)

        interpreter.set_tensor(
            input_index,
            sample
        )

        interpreter.invoke()

        output = interpreter.get_tensor(output_index)

        if output_details[0]["dtype"] == np.int8:
            out_scale, out_zero = output_details[0]["quantization"]
            output = (output.astype(np.float32) - out_zero) * out_scale

        predictions.append(int(np.argmax(output)))

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    ######################################################
    # Model Size
    ######################################################

    model_size = os.path.getsize(
        model_path
    ) / 1024

    ######################################################
    # Memory
    ######################################################

    memory_mb = memory_after / (1024 * 1024)

    ######################################################
    # Return Metrics
    ######################################################

    return calculate_metrics(

        latencies=latencies,

        accuracy=accuracy,

        model_size_kb=model_size,

        cpu_percent=cpu_percent,

        memory_mb=memory_mb,

        variant=variant

    )

##########################################################
# Execute Benchmarks
##########################################################

results = []

##########################################################
# M1 - FP32
##########################################################

if M1_WEIGHTS.exists():

    results.append(

        benchmark_keras(

            M1_WEIGHTS,

            "M1_FP32"

        )

    )

else:

    print()

    print("M1 weights not found")

    print(M1_WEIGHTS)

##########################################################
# M2 - PTQ INT8
##########################################################

if M2_MODEL.exists():

    results.append(

        benchmark_tflite(

            M2_MODEL,

            "M2_PTQ_INT8"

        )

    )

else:

    print()

    print("M2 model not found")

    print(M2_MODEL)

##########################################################
# M3 - Pruned PTQ INT8
##########################################################

if M3_MODEL.exists():

    results.append(

        benchmark_tflite(

            M3_MODEL,

            "M3_PRUNED_PTQ_INT8"

        )

    )

else:

    print()

    print("M3 model not found")

    print(M3_MODEL)

##########################################################
# Results DataFrame
##########################################################

benchmark_df = pd.DataFrame(results)

benchmark_df = benchmark_df[[

    "Variant",

    "Mean Latency (ms)",

    "P95 Latency (ms)",

    "Model Size (KB)",

    "Accuracy (%)",

    "Energy (mJ)",

    "CPU (%)",

    "Memory (MB)"

]]

##########################################################
# Save CSV
##########################################################

csv_file = RESULTS_DIR / "benchmark_results.csv"

benchmark_df.to_csv(

    csv_file,

    index=False

)

##########################################################
# Console Output
##########################################################

print()

print("=" * 75)

print("FINAL BENCHMARK RESULTS")

print("=" * 75)

print()

print(benchmark_df.to_string(index=False))

print()

print("=" * 75)

print("Benchmark results saved")

print(csv_file)

print("=" * 75)

print()

##########################################################
# System Information
##########################################################

print("System Information")

print("-" * 75)

print("Operating System :", platform.platform())

print("Processor        :", platform.processor())

print("Python           :", platform.python_version())

print("TensorFlow       :", tf.__version__)



##########################################################
# Recommended Variant
##########################################################

best = benchmark_df.sort_values(

    by=[

        "Accuracy (%)",

        "Mean Latency (ms)"

    ],

    ascending=[False, True]

).iloc[0]

print()

print("=" * 75)

print("Recommended Variant")

print("=" * 75)

print(

    f"{best['Variant']} "

    f"(Accuracy={best['Accuracy (%)']}%, "

    f"Latency={best['Mean Latency (ms)']} ms)"

)