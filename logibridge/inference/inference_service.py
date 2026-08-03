#!/usr/bin/env python3
"""
----------------------------------------------------------
LogiBridge V3
Component: Edge Inference Service

Responsibilities:
1. Subscribe to MQTT sensor topics (temperature, vibration).
2. Maintain sliding windows of sensor data.
3. Use data_pipeline.preprocessing to get normalized feature vectors.
4. Dynamically quantize inputs to INT8 and run TFLite inference.
5. Dequantize output probabilities for confidence calculation.
6. Implement sustained alert logic (N consecutive 'Critical' predictions).
7. Log every inference and alert to a local SQLite database.
8. Publish inference results to an MQTT topic for monitoring.
----------------------------------------------------------
"""
import os
import sys
import json
import time
import sqlite3
from collections import deque
from pathlib import Path
import argparse

# --- Relative Path Setup ---
ROOT = Path(__file__).resolve().parents[1]
MONITORING_DIR = ROOT / "monitoring"

if str(MONITORING_DIR) not in sys.path:
    sys.path.insert(0, str(MONITORING_DIR))

import drift_monitor


str_root = str(ROOT)
if str_root not in sys.path:
    sys.path.insert(0, str_root)
os.chdir(str_root)

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf

# Standard module import from project structure
from data_pipeline import preprocessing

parser = argparse.ArgumentParser()

parser.add_argument(
    "--build-reference",
    action="store_true",
    help="Generate PSI reference distribution"
)

args = parser.parse_args()

# Portable Relative File Paths
STATS_PATH = ROOT / "data_pipeline" / "training_stats.npy"
MODEL_PATH = ROOT / "models" / "m2_ptq_int8.tflite"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "inference" / "model.tflite"

DATABASE_PATH = ROOT / "monitoring" / "alerts.db"

# Initialize Preprocessing Stats
preprocessing.load_stats(str(STATS_PATH))
if not args.build_reference:
    drift_monitor.load_reference()




# --- Configuration ---
BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))
TRUCK_ID = os.getenv("TRUCK_ID", "TRUCK01")

# MQTT Topics
TEMP_TOPIC = f"logibridge/trucks/{TRUCK_ID}/temperature"
VIB_TOPIC = f"logibridge/trucks/{TRUCK_ID}/vibration"
INFERENCE_TOPIC = f"logibridge/trucks/{TRUCK_ID}/inference"

# Processing Parameters
WINDOW_TEMP_SAMPLES = 30  # 30 seconds @ 1Hz
WINDOW_VIB_SAMPLES = 15   # 30 seconds @ 0.5Hz
STEP_SECONDS = 1 #10         # Process window every 10 seconds
ALERT_THRESHOLD = 3       # N consecutive criticals to fire alert

# --- Global Buffers & State ---
temperature_buffer = deque(maxlen=WINDOW_TEMP_SAMPLES)
vibration_buffer = deque(maxlen=WINDOW_VIB_SAMPLES)
consecutive_critical_deque = deque(maxlen=ALERT_THRESHOLD)
last_processing_time = 0

# --- SQLite Database Setup ---
def initialize_database():
    """Create SQLite database and alerts table with WAL mode."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            truck_id TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            alert_fired INTEGER NOT NULL
        )
    """)
    conn.commit()
    print(f"SQLite database ready at '{DATABASE_PATH}'")
    return conn, cursor

# --- TFLite Model Loading ---
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"TFLite model not found at '{MODEL_PATH}'")

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
print(f"TFLite model loaded from '{MODEL_PATH}'")

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker.")
        client.subscribe(TEMP_TOPIC, qos=1)
        client.subscribe(VIB_TOPIC, qos=1)
    else:
        print(f"Failed to connect to MQTT, return code {rc}")

def on_message(client, userdata, msg):
    """Append incoming sensor data to the appropriate buffer."""
    try:
        payload = json.loads(msg.payload.decode())
        value = float(payload["value"])
        if msg.topic == TEMP_TOPIC:
            temperature_buffer.append(value)
        elif msg.topic == VIB_TOPIC:
            vibration_buffer.append(value)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing MQTT message: {e}")

# --- Main Service Logic ---
if __name__ == "__main__":
    db_conn, db_cursor = initialize_database()
    
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        mqtt_client = mqtt.Client()

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(BROKER, PORT, 60)
    
    print("Inference service started. Waiting for sensor data...")
    mqtt_client.loop_start()

    try:
        while True:
            if (time.time() - last_processing_time) < STEP_SECONDS:
                time.sleep(0.5)
                continue
            
            last_processing_time = time.time()

            if len(temperature_buffer) < WINDOW_TEMP_SAMPLES or len(vibration_buffer) < WINDOW_VIB_SAMPLES:
                continue

            # --- Preprocessing ---
            window_data = {
                'temperature': list(temperature_buffer),
                'vibration': list(vibration_buffer)
            }
            
            normalized_features = preprocessing.preprocess(window_data)

            # --- Quantization & Inference ---
            input_data = normalized_features.reshape(1, -1)

            if input_details['dtype'] == np.int8:
                scale, zero_point = input_details['quantization']
                if scale > 0:
                    input_data = np.round(input_data / scale + zero_point).astype(np.int8)
                else:
                    input_data = input_data.astype(np.int8)
            else:
                input_data = input_data.astype(np.float32)

            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details['index'])[0]

            # Dequantize output probabilities
            if output_details['dtype'] == np.int8:
                scale, zero_point = output_details['quantization']
                if scale > 0:
                    output_data = (output_data.astype(np.float32) - zero_point) * scale

            # --- Post-processing ---
            
            # ----------------------------------------------------
            # # Convert output to probabilities
            # # ----------------------------------------------------

            if 0.99 <= output_data.sum() <= 1.01:

                probabilities = output_data.astype(np.float32)

            else:

                exp_scores = np.exp(output_data - np.max(output_data))
                probabilities = exp_scores / np.sum(exp_scores)

            prediction_idx = int(np.argmax(probabilities))

            confidence = float(probabilities[prediction_idx])
            if args.build_reference:
                drift_monitor.add_reference_confidence(confidence)
            else:
                drift_monitor.add_confidence(confidence)

            labels = {
                0: "Normal",
                1: "Warning",
                2: "Critical"
            }

            prediction_label = labels[prediction_idx]

            # --- Sustained Alert Logic ---
            consecutive_critical_deque.append(prediction_label == "Critical")
            alert_fired = (len(consecutive_critical_deque) == ALERT_THRESHOLD and all(consecutive_critical_deque))

            # --- Logging and Publishing ---
            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {prediction_label:<8} (Conf: {confidence:.2f})")
            if alert_fired:
                print("\n" + "="*50)
                print(f"*** CRITICAL ALERT FOR TRUCK {TRUCK_ID} ***")
                print(f"*** {ALERT_THRESHOLD} CONSECUTIVE CRITICAL EVENTS DETECTED ***")
                print("="*50 + "\n")

            # Log to SQLite
            db_cursor.execute(
                "INSERT INTO alerts (truck_id, prediction, confidence, alert_fired) VALUES (?, ?, ?, ?)",
                (TRUCK_ID, prediction_label, confidence, int(alert_fired))
            )
            db_conn.commit()
            
            # Publish to MQTT for monitoring
            mqtt_client.publish(INFERENCE_TOPIC, json.dumps({
                "truck_id": TRUCK_ID,
                "prediction": prediction_label,
                "confidence": confidence,
                "alert_fired": alert_fired
            }))

    except KeyboardInterrupt:
        print("\nStopping inference service.")
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        db_conn.close()


    