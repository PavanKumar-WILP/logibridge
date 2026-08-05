#!/usr/bin/env python3

"""
-------------------------------------------------------
LogiBridge

Component G

Benchmarking

Task F2

Measures

1. Mean inference latency
2. P95 latency
3. Model size
4. Classification accuracy
5. Energy per inference

Additional Metrics

- CPU Usage
- Memory Usage
- System Information

-------------------------------------------------------
"""

import json
import os
import platform
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

warnings.simplefilter("ignore")

# ------------------------------------------------------
# Paths
# ------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "inference" / "model.tflite"

if not MODEL.exists():

    raise FileNotFoundError(
        f"{MODEL} not found.\n"
        "Run convert_ptq.py first."
    )

RESULTS = ROOT / "optimisation" / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_FILE = ROOT / "training" / "models" / "model_metrics.json"

# ------------------------------------------------------
# Load Accuracy
# ------------------------------------------------------

accuracy = "N/A"

if METRICS_FILE.exists():

    with open(METRICS_FILE) as f:

        accuracy = json.load(f)["accuracy"] * 100

# ------------------------------------------------------
# Load Model
# ------------------------------------------------------

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

print("=" * 60)
print("TensorFlow Lite Benchmark")
print("=" * 60)
print("Model :", MODEL)
print()

# ------------------------------------------------------
# Dummy Input
# ------------------------------------------------------

if input_details[0]["dtype"] == np.int8:

    dummy = np.random.randint(
        -128,
        127,
        size=(1, 6),
        dtype=np.int8
    )

else:

    dummy = np.random.rand(
        1,
        6
    ).astype(np.float32)

# ------------------------------------------------------
# Warm-up
# ------------------------------------------------------

print("Running warm-up...")

for _ in range(10):

    interpreter.set_tensor(
        input_details[0]["index"],
        dummy
    )

    interpreter.invoke()

    interpreter.get_tensor(
        output_details[0]["index"]
    )

print("Warm-up complete.\n")

# ------------------------------------------------------
# Benchmark
# ------------------------------------------------------

TOTAL_RUNS = 200

latencies = []

print(f"Running {TOTAL_RUNS} benchmark iterations...\n")

process = psutil.Process()

cpu_before = psutil.cpu_percent(interval=1)

memory_before = process.memory_info().rss

for _ in range(TOTAL_RUNS):

    start = time.perf_counter()

    interpreter.set_tensor(
        input_details[0]["index"],
        dummy
    )

    interpreter.invoke()

    interpreter.get_tensor(
        output_details[0]["index"]
    )

    end = time.perf_counter()

    latencies.append(
        (end - start) * 1000
    )

cpu_after = psutil.cpu_percent(interval=1)

memory_after = process.memory_info().rss

# ------------------------------------------------------
# Calculate Metrics
# ------------------------------------------------------

latency_mean = np.mean(latencies)

latency_std = np.std(latencies)

latency_p95 = np.percentile(
    latencies,
    95
)

throughput = (
    1000.0 / latency_mean
    if latency_mean > 0
    else 0
)

model_size_kb = (
    os.path.getsize(MODEL)
    / 1024
)

memory_mb = (
    memory_after
    / (1024 * 1024)
)

# ------------------------------------------------------
# Energy Estimation
# ------------------------------------------------------
#
# Assignment Formula:
#
#     E = P × t
#
# where
#
# P = CPU Power (Watts)
# t = inference time (seconds)
#
# We assume a 15W laptop CPU.
#
# Energy is reported in mJ.
# ------------------------------------------------------

CPU_TDP_WATTS = 15.0

energy_mj = (
    CPU_TDP_WATTS
    * (latency_mean / 1000.0)
    * 1000
)

# ------------------------------------------------------
# Console Summary
# ------------------------------------------------------

print("=" * 60)
print("Benchmark Results")
print("=" * 60)

print(f"Runs                    : {TOTAL_RUNS}")
print(f"Mean Latency (ms)       : {latency_mean:.6f}")
print(f"P95 Latency (ms)        : {latency_p95:.6f}")
print(f"Latency Std (ms)        : {latency_std:.6f}")
print(f"Throughput (inf/sec)    : {throughput:.2f}")
print(f"CPU Usage (%)           : {cpu_after:.2f}")
print(f"Memory (MB)             : {memory_mb:.2f}")
print(f"Model Size (KB)         : {model_size_kb:.3f}")

if accuracy != "N/A":

    print(f"Accuracy (%)            : {accuracy:.2f}")

else:

    print("Accuracy (%)            : N/A")

print(f"Energy / Inference (mJ) : {energy_mj:.4f}")

print()

# ------------------------------------------------------
# Required Assignment Metrics
# ------------------------------------------------------

required_df = pd.DataFrame({

    "Metric":[

        "Mean Latency (ms)",

        "P95 Latency (ms)",

        "Model Size (KB)",

        "Classification Accuracy (%)",

        "Energy per Inference (mJ)"

    ],

    "Value":[

        latency_mean,

        latency_p95,

        model_size_kb,

        accuracy,

        energy_mj

    ]

})

# ------------------------------------------------------
# Additional Metrics
# ------------------------------------------------------

extra_df = pd.DataFrame({

    "Metric":[

        "Latency Std (ms)",

        "Throughput (inf/sec)",

        "CPU Usage (%)",

        "Process Memory (MB)"

    ],

    "Value":[

        latency_std,

        throughput,

        cpu_after,

        memory_mb

    ]

})

benchmark_df = pd.concat(
    [required_df, extra_df],
    ignore_index=True
)

print(benchmark_df)

# ------------------------------------------------------
# Save Benchmark Results
# ------------------------------------------------------

csv_file = RESULTS / "benchmark_results.csv"

benchmark_df.to_csv(
    csv_file,
    index=False
)

print()
print(f"Benchmark results saved to:")
print(csv_file)

# ------------------------------------------------------
# Final Summary
# ------------------------------------------------------

print()
print("=" * 60)
print("Benchmark Completed")
print("=" * 60)

print()
print("System Information")
print("------------------")
print("OS        :", platform.platform())
print("Processor :", platform.processor())
print("Python    :", platform.python_version())
print("TensorFlow:", tf.__version__)

print()
print("Generated Files")
print("----------------")
print(csv_file)

print()
print("=" * 60)
print("Required Assignment Metrics")
print("=" * 60)
print(f"Mean Latency (200 runs)      : {latency_mean:.6f} ms")
print(f" P95 Latency                  : {latency_p95:.6f} ms")
print(f" Model Size                   : {model_size_kb:.3f} KB")

if accuracy != "N/A":
    print(f" Classification Accuracy      : {accuracy:.2f} %")
else:
    print(" Classification Accuracy      : N/A (model_metrics.json not found)")

print(f" Energy per Inference         : {energy_mj:.4f} mJ")

print()
print("Additional Metrics")
print("------------------")
print(f"Latency Std        : {latency_std:.6f} ms")
print(f"Throughput         : {throughput:.2f} inf/sec")
print(f"CPU Usage          : {cpu_after:.2f} %")
print(f"Process Memory     : {memory_mb:.2f} MB")

print()
print("Benchmark finished successfully.")
