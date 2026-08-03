# Constraint Analysis

## Project Scenario

LogiBridge is an Edge AI solution developed for refrigerated freight transportation. The objective is to continuously monitor environmental conditions inside refrigerated trucks while maintaining reliable operation even in areas with intermittent or unavailable network connectivity.

The system performs local data processing and inference on the truck, allowing immediate detection of abnormal operating conditions without relying on cloud infrastructure.

---

# Functional Constraints

The system must continuously monitor:

- Temperature
- Vibration
- Door status

Sensor measurements are published through a local MQTT broker and processed by the edge inference engine.

The trained AI model classifies the current operating condition into one of three categories:

- Normal
- Warning
- Critical

Critical events must immediately generate alerts.

---

# Edge Computing Constraints

Unlike cloud-based monitoring systems, inference must execute entirely on the edge device.

Requirements include:

- Low inference latency
- Small memory footprint
- Minimal CPU utilisation
- Offline operation
- Local alert persistence

The deployed TensorFlow Lite model satisfies these constraints through model optimisation using post-training quantization and pruning.

---

# Connectivity Constraints

Long-distance freight vehicles frequently encounter unreliable cellular coverage.

The system therefore supports:

- Continuous offline inference
- Local SQLite alert storage
- Automatic synchronization after connectivity restoration

This prevents alert loss during communication failures.

---

# Computational Constraints

The target hardware provides limited processing capability compared with cloud servers.

The Edge AI model must therefore:

- Execute within milliseconds
- Consume limited RAM
- Maintain high classification accuracy
- Support TensorFlow Lite deployment

These requirements motivated the use of a lightweight Multi-Layer Perceptron (MLP).

---

# Reliability Constraints

Cold-chain logistics require continuous monitoring because temperature excursions may damage pharmaceuticals, vaccines or perishable food.

The system therefore provides:

- Real-time inference
- Immediate alert generation
- Persistent local storage
- Drift monitoring
- Automated deployment

These mechanisms improve operational reliability.

---

# Data Quality Constraints

Sensor measurements may contain:

- Noise
- Missing observations
- Short spikes
- Sensor drift

The preprocessing pipeline mitigates these issues using:

- Moving-average filtering
- Sliding windows
- Feature extraction
- Feature normalization

This improves model robustness.

---

# Security Constraints

Communication between software components uses MQTT within the local edge network.

Only processed alerts are synchronized to the operations centre, reducing unnecessary communication overhead and limiting exposure of raw sensor data.

---

# Conclusion

The proposed LogiBridge architecture satisfies the operational, computational, networking and reliability constraints associated with cold-chain transportation by combining Edge AI, MQTT communication, TensorFlow Lite inference and offline-first operation.