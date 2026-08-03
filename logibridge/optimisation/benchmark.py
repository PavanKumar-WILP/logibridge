#!/usr/bin/env python3

"""
-------------------------------------------------------

Component G

Benchmark

Measures

Latency
CPU
Memory
Model Size

-------------------------------------------------------
"""

import os
import time
from pathlib import Path
import platform

import numpy as np
import psutil
import tensorflow as tf

import warnings

warnings.simplefilter("ignore")

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


interpreter = tf.lite.Interpreter(

    model_path=str(MODEL)

)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

if input_details[0]["dtype"] == np.int8:

    scale, zero = input_details[0]["quantization"]

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


latencies = []

cpu_before = psutil.cpu_percent()

for _ in range(1000):

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

        (end-start)*1000

    )

cpu_after = psutil.cpu_percent()


latency = np.mean(latencies)
latency_std = np.std(latencies)

latency_p95 = np.percentile(

    latencies,

    95

)

throughput = 1000.0 / latency if latency > 0 else 0

model_size = os.path.getsize(MODEL)/1024

cpu = cpu_after

process = psutil.Process()
memory_mb = process.memory_info().rss / (1024 * 1024)


import pandas as pd

df = pd.DataFrame({

    "Metric": [

        "Average Latency (ms)",

        "P95 Latency (ms)",

        "Latency Std (ms)",

        "Throughput (inf/sec)",

        "CPU (%)",

        "Process Memory (RSS) (MB)",

        "Model Size (KB)"

    ],

    "Value": [

        latency,

        latency_p95,

        latency_std,

        throughput,

        cpu,

        memory_mb,

        model_size

    ]

})
csv_file = RESULTS / "benchmark_results.csv"

df.to_csv(

    csv_file,

    index=False

)

print(df)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(

    df["Metric"],

    df["Value"]

)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    RESULTS/"pareto_chart.png",

    dpi=300

)

plt.close()

print()

print("Benchmark completed.")

print()

print("System")

print(platform.platform())

print("Processor")

print(platform.processor())