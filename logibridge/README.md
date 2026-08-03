# LogiBridge – Edge AI Cold Chain Monitoring System

## Project Overview

LogiBridge is an Edge AI-based cold-chain monitoring system designed for refrigerated transport vehicles. The system continuously monitors temperature, vibration, and door events using IoT sensors and performs real-time inference directly on the edge device.

Unlike cloud-only monitoring systems, LogiBridge continues to operate even during network outages by performing local inference, storing alerts locally, and synchronizing them with the FreightBridge Operations Centre once connectivity is restored.

The project demonstrates the complete Edge AI lifecycle including data collection, preprocessing, model training, model optimisation, deployment, monitoring, and benchmarking.

---

# Repository Structure

```
logibridge/
├── README.md
├── scenario_architecture/
│   ├── constraint_analysis.md
│   └── system_architecture.png
├── hardware/
│   └── hardware_justification.md
├── data_pipeline/
│   ├── simulator.py
│   ├── preprocessing.py
│   ├── training_stats.npy
│   └── mqtt_architecture.md
├── training/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── convert_ptq.py
│   ├── prune_quantise.py
│   └── models/
├── inference/
│   ├── Dockerfile
│   ├── inference_service.py
│   └── model.tflite
├── monitoring/
│   ├── drift_monitor.py
│   └── reference_dist.json
├── deployment/
│   └── logibridge_deploy.yml
├── optimisation/
│   ├── benchmark.py
│   └── results/
│       ├── benchmark_results.csv
│       └── pareto_chart.png
```

---

# System Architecture

The complete system consists of the following modules:

1. Sensor Simulator
2. MQTT Communication Layer
3. Data Preprocessing Pipeline
4. Edge AI Inference Engine
5. Alert Management
6. Drift Monitoring
7. Deployment Automation
8. Performance Benchmarking

The complete architecture is available in

```
scenario_architecture/system_architecture.png
```

---

# Features

- Real-time temperature monitoring
- Real-time vibration monitoring
- Door event monitoring
- MQTT communication
- Sliding window preprocessing
- Moving average filtering
- Feature extraction
- Feature normalization
- TensorFlow MLP classifier
- TensorFlow Lite inference
- Offline inference
- Local alert storage
- Alert synchronization
- Drift monitoring using Population Stability Index (PSI)
- Model optimisation using pruning and quantization
- Docker deployment
- Automated benchmarking

---

# Data Pipeline

The preprocessing pipeline performs the following operations:

1. Moving Average Filtering
2. Sliding Window Generation
3. Temperature Feature Extraction
4. Vibration Feature Extraction
5. Feature Normalization
6. Feature Vector Generation

Final Feature Vector

| Feature |
|----------|
| Temperature Mean |
| Temperature Standard Deviation |
| Temperature Rate of Change |
| Vibration RMS |
| Vibration Peak |
| Vibration Kurtosis |

---

# Machine Learning Model

Model Type

- Multi-Layer Perceptron (MLP)

Input Features

- 6

Output Classes

| Class | Description |
|--------|-------------|
| 0 | Normal |
| 1 | Warning |
| 2 | Critical |

Training uses

- Adam Optimizer
- Early Stopping
- Validation Split
- TensorFlow Keras

---

# Model Optimisation

The following optimisation techniques are implemented.

## Post Training Quantization

FP32

↓

TensorFlow Lite

## Model Pruning

- Polynomial Decay
- Fine Tuning
- TensorFlow Model Optimization Toolkit

---

# Deployment

Deployment is automated using

- Docker
- YAML Deployment Script

The inference service performs

- MQTT Subscription
- Feature Extraction
- Normalization
- TensorFlow Lite Inference
- Alert Logging
- Offline Operation

---

# Monitoring

The monitoring module performs

- Population Stability Index (PSI)
- Feature Drift Detection
- Drift Report Generation

---

# Benchmarking

Performance metrics include

- Inference Latency
- Throughput
- CPU Utilisation
- Memory Consumption
- Model Size

Benchmark results are stored in

```
optimisation/results/
```

---

# Software Requirements

- Python 3.11+
- TensorFlow
- TensorFlow Lite
- TensorFlow Model Optimization Toolkit
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Paho MQTT
- Mosquitto MQTT Broker

---

# Installation

Install dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn tensorflow tensorflow-model-optimization paho-mqtt psutil
```

---

# Running the Project

## Step 1

Start Mosquitto

```bash
mosquitto
```

---

## Step 2

Generate Dataset

```bash
python training/generate_dataset.py
```

---

## Step 3

Train Model

```bash
python training/train_model.py
```

---

## Step 4

Convert Model

```bash
python training/convert_ptq.py
```

---

## Step 5

Run Inference

```bash
python inference/inference_service.py
```

---

## Step 6

Run Drift Monitoring

```bash
python monitoring/drift_monitor.py
```

---

## Step 7

Benchmark

```bash
python optimisation/benchmark.py
```

---

# Expected Outputs

The project generates the following outputs.

```
training/
    dataset.csv

data_pipeline/
    training_stats.npy

training/models/
    model.keras

inference/
    model.tflite

monitoring/
    alerts.db
    reference_dist.json
    drift_report.json

optimisation/results/
    benchmark_results.csv
    pareto_chart.png
```

---

# Authors
	Pavan Kumar S
---

# License

This project is submitted for academic evaluation.