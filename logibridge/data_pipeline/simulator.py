#!/usr/bin/env python3

"""
-------------------------------------------------------
LogiEdge Sensor Simulator
Component C - Task C1

Generates three MQTT sensor streams

1. Temperature (1 Hz)
2. Vibration RMS (0.5 Hz)
3. Door Events (Discrete)

Supports anomaly modes

none
temp_drift
vibration
combined
-------------------------------------------------------
"""

import argparse
import json
import random
import time
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt


# ----------------------------------------------------
# MQTT Configuration
# ----------------------------------------------------

BROKER = "localhost"
PORT = 1883
KEEPALIVE = 60

TRUCK_ID = "TRUCK01"

TOPIC_TEMP = f"logibridge/trucks/{TRUCK_ID}/temperature"
TOPIC_VIB = f"logibridge/trucks/{TRUCK_ID}/vibration"
TOPIC_DOOR = f"logibridge/trucks/{TRUCK_ID}/door"



# ----------------------------------------------------
# Simulator Parameters
# ----------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SIMULATOR = ROOT / "data_pipeline" / "simulator.py"

DATASET_FILE = ROOT / "training" / "dataset.csv"

TRAINING_STATS = ROOT / "data_pipeline" / "training_stats.npy"

TEMP_SETPOINT = 4.0

TEMP_MEAN = 4.0
TEMP_STD = 0.3

TEMP_DRIFT = 0.08

VIB_MEAN = 0.45
VIB_STD = 0.05

VIB_ANOMALY_MEAN = 1.20
VIB_ANOMALY_STD = 0.15


# ----------------------------------------------------
# MQTT Client
# ----------------------------------------------------

client = mqtt.Client()

client.connect(BROKER, PORT, KEEPALIVE)
client.loop_start()


# ----------------------------------------------------
# Command Line
# ----------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--anomaly",
    choices=[
        "none",
        "temp_drift",
        "vibration",
        "combined"
    ],
    default="none",
    help="Select anomaly mode"
)

args = parser.parse_args()

print("=" * 60)
print("LogiEdge Sensor Simulator")
print(f"Truck ID : {TRUCK_ID}")
print(f"Mode     : {args.anomaly}")
print("=" * 60)


# ----------------------------------------------------
# Temperature State
# ----------------------------------------------------

current_temperature = TEMP_SETPOINT

last_vibration_publish = time.time()


# ----------------------------------------------------
# Main Loop
# ----------------------------------------------------
try:
    while True:

        timestamp = datetime.now().isoformat()

        # ==================================================
        # Temperature
        # ==================================================

        if args.anomaly in ["temp_drift", "combined"]:

            current_temperature = min(
            current_temperature + TEMP_DRIFT,
                12.0)

            temperature = current_temperature + np.random.normal(
                0,
                0.10
            )

        else:

            current_temperature = np.random.normal(
                TEMP_MEAN,
                TEMP_STD
            )

            temperature = current_temperature

        temp_payload = {

            "truck_id": TRUCK_ID,

            "sensor": "temperature",

            "timestamp": timestamp,

            "value": round(float(temperature), 2)

        }

        result = client.publish(

            TOPIC_TEMP,

            json.dumps(temp_payload),

            qos=1

        )
        result.wait_for_publish()

        print(f"TEMP : {temperature:.2f} °C")

        # ==================================================
        # Vibration (0.5 Hz)
        # ==================================================

        if time.time() - last_vibration_publish >= 2:

            last_vibration_publish = time.time()

            if args.anomaly in ["vibration", "combined"]:

                vibration = np.random.normal(

                    VIB_ANOMALY_MEAN,

                    VIB_ANOMALY_STD

                )

            else:

                vibration = np.random.normal(

                    VIB_MEAN,

                    VIB_STD

                )

            vib_payload = {

                "truck_id": TRUCK_ID,
                "sensor": "vibration",
                "timestamp": timestamp,
                "value": round(float(vibration), 3)

            }

            result = client.publish(

                TOPIC_VIB,

                json.dumps(vib_payload),

                qos=1

            )
            result.wait_for_publish()

            print(f"VIB  : {vibration:.3f} g")

        # ==================================================
        # Door Events
        # ==================================================

        if random.random() < 0.02:

            event = random.choice(

                [

                    "OPEN",

                    "CLOSE"

                ]

            )

            door_payload = {

                "truck_id": TRUCK_ID,
                "sensor": "door",
                "timestamp": timestamp,
                "event": event

            }

            result = client.publish(
                TOPIC_DOOR,
                json.dumps(door_payload),
                qos=1,
                retain=False
            )

            result.wait_for_publish()

            print(f"DOOR : {event}")

        time.sleep(1)
except KeyboardInterrupt:
    print("\nSimulator stopped by user")
    client.loop_stop()
    client.disconnect()