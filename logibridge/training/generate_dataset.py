#!/usr/bin/env python3
"""
---------------------------------------------------------
LogiEdge
Component D - Dataset Generation

This script:

1. Starts simulator automatically
2. Subscribes to MQTT topics
3. Collects sensor readings
4. Creates sliding windows
5. Extracts features
6. Saves dataset.csv
7. Computes training_stats.npy
---------------------------------------------------------
"""

import argparse
import csv
import json
import os
from pyexpat import features
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

# Import preprocessing functions
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_pipeline.preprocessing import (
    extract_features
)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

BROKER = "localhost"
PORT = 1883

TRUCK_ID = "TRUCK01"

TEMP_TOPIC = f"logibridge/trucks/{TRUCK_ID}/temperature"
VIB_TOPIC = f"logibridge/trucks/{TRUCK_ID}/vibration"

WINDOW_SECONDS = 30
STEP_SECONDS =  10

TEMP_HZ = 1           # 1 sample/second
VIB_HZ = 0.5          # 1 sample every 2 seconds

TEMP_BUFFER_SIZE = WINDOW_SECONDS * TEMP_HZ      # 30
VIB_BUFFER_SIZE = int(WINDOW_SECONDS * VIB_HZ)   # 15

ROOT = Path(__file__).resolve().parents[1] 
SIMULATOR = ROOT / "data_pipeline" / "simulator.py" 
DATASET_FILE = ROOT / "training" / "dataset.csv" 
TRAINING_STATS = ROOT / "data_pipeline" / "training_stats.npy"

NORMAL_DURATION = 20 * 60
WARNING_DURATION = 15 * 60
CRITICAL_DURATION = 15 * 60

DRIFT_SECONDS = 25  # Extra time to allow for drift in the simulator
# ----------------------------------------------------
# Global Buffers
# ----------------------------------------------------

temperature_buffer = deque(maxlen=TEMP_BUFFER_SIZE)

vibration_buffer = deque(maxlen=VIB_BUFFER_SIZE)

dataset = []

current_label = 0

last_feature_time = time.time()

simulator_process = None

# ----------------------------------------------------
# MQTT Callbacks
# ----------------------------------------------------

def on_connect(client, userdata, flags, rc):

    print("Connected to MQTT Broker")
    global mqtt_connected
    mqtt_connected = True
    r1 = client.subscribe(TEMP_TOPIC, qos=1)
    r2 = client.subscribe(VIB_TOPIC, qos=1)

    print(r1)
    print(r2)


def on_message(client, userdata, msg):

    global temperature_buffer
    global vibration_buffer

    print("Received:", msg.topic)

    payload = json.loads(msg.payload.decode())

    if msg.topic == TEMP_TOPIC:

        temperature_buffer.append(payload["value"])

    elif msg.topic == VIB_TOPIC:

        vibration_buffer.append(payload["value"])

    last_message_time = time.time()

        
        
# ----------------------------------------------------
# Simulator Control
# ----------------------------------------------------

def start_simulator(mode):

    global simulator_process

    print(f"\nStarting simulator ({mode})")

    simulator_process = subprocess.Popen(
        [
            sys.executable,
            str(SIMULATOR),
            "--anomaly",
            mode
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def stop_simulator():

    global simulator_process

    if simulator_process is not None:

        simulator_process.terminate()

        simulator_process.wait()

        simulator_process = None
        

# ----------------------------------------------------
# Dataset Collection
# ----------------------------------------------------

def collect_features(label):

    global dataset
    global last_feature_time

    if len(temperature_buffer) < TEMP_BUFFER_SIZE:
        return

    if len(vibration_buffer) < VIB_BUFFER_SIZE:
        return

    now = time.time()

    if now - last_feature_time < STEP_SECONDS:
        return

    last_feature_time = now

    temp_window = np.array(temperature_buffer)

    vib_window = np.array(vibration_buffer)

    print("\n========================")
    print("Temperature buffer length:", len(temp_window))
    print("Vibration buffer length :", len(vib_window))

    print("Temperature Window:")
    print(temp_window)

    print("Vibration Window:")
    print(vib_window)

    features = extract_features(
        temp_window,
        vib_window
    )
    print("Extracted Features:")
    print(features)
    print("========================")
    row = features.tolist()

    row.append(label)

    dataset.append(row)

    print(
        f"Collected sample #{len(dataset)} "
        f"Label={label}"
    )

    print(
        f"\rSamples={len(dataset):5d} | Label={label} \n",
        end="",
        flush=True
    )
    
# ----------------------------------------------------
# MQTT Client
# ----------------------------------------------------

client = None


# ----------------------------------------------------
# Run one collection phase
# ----------------------------------------------------
def run_phase(mode, duration, label):

    global temperature_buffer
    global vibration_buffer

    print(f"\n{'='*60}")
    print(f"Mode     : {mode}")
    print(f"Label    : {label}")
    print(f"Duration : {duration} sec")
    print(f"{'='*60}")

    temperature_buffer.clear()
    vibration_buffer.clear()

    start_simulator(mode)

    time.sleep(5)

    start = time.time()

    initial = len(dataset)

    try:

        while time.time() - start < duration + DRIFT_SECONDS:

            collect_features(label)

            time.sleep(0.25)

    finally:

        stop_simulator()

    print()

    print(
        f"Collected {len(dataset)-initial} samples."
    )

# ----------------------------------------------------
# Save Dataset
# ----------------------------------------------------
def save_dataset():

    DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    header = [

        "temp_mean",
        "temp_std",
        "temp_rate",
        "vib_rms",
        "vib_peak",
        "vib_kurtosis",
        "label"

    ]

 
    with open(DATASET_FILE, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(dataset)

    print(f"\nAdded {len(dataset)} samples to {DATASET_FILE}")    
# ----------------------------------------------------
# Compute Training Statistics
# ----------------------------------------------------

def save_training_statistics():

    normal_rows = [

        row[:-1]

        for row in dataset

        if row[-1] == 0

    ]

    if len(normal_rows) == 0:

        raise ValueError(

            "No normal samples available to compute training statistics. "

            "Collect at least one normal sample before saving stats."

        )

    normal = np.atleast_2d(np.array(normal_rows, dtype=float))

    mean = normal.mean(axis=0)

    std = normal.std(axis=0)

    std[std == 0] = 1

    stats = {

        "mean": mean,

        "std": std

    }

    np.save(

        TRAINING_STATS,

        stats

    )

    print(

        f"Training statistics saved to "

        f"{TRAINING_STATS}"

    )
    
# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():

    global client
    global mqtt_connected
    global dataset

    mqtt_connected = False

    # Start with a fresh dataset every run
    dataset.clear()

    # Remove old dataset if it exists
    if DATASET_FILE.exists():
        DATASET_FILE.unlink()

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)

    print("Waiting for MQTT connection...")

    client.loop_start()

    while not mqtt_connected:
        time.sleep(0.1)

    print("Connected to MQTT Broker")

    try:

        # -------------------------------------------------
        # Phase 1 : Normal
        # -------------------------------------------------
        run_phase(
            mode="none",
            duration=NORMAL_DURATION,
            label=0
        )

        # -------------------------------------------------
        # Phase 2 : Warning
        # -------------------------------------------------
        run_phase(
            mode="temp_drift",
            duration=WARNING_DURATION,
            label=1
        )

        # -------------------------------------------------
        # Phase 3 : Critical
        # -------------------------------------------------
        run_phase(
            mode="combined",
            duration=CRITICAL_DURATION,
            label=2
        )

        print("\nDataset collection completed.")

        save_dataset()

        save_training_statistics()

        print("\nAll files generated successfully.")

    finally:

        stop_simulator()

        client.loop_stop()

        client.disconnect()

if __name__ == "__main__":

    main()
    
    
