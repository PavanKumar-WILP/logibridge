# MQTT Communication Architecture

## Overview

LogiBridge uses the MQTT publish-subscribe messaging protocol to exchange sensor data between software components.

MQTT provides lightweight communication with minimal bandwidth requirements, making it well suited for resource-constrained edge devices.

---

# Architecture

```
Temperature Sensor
        │
        ▼
Vibration Sensor
        │
        ▼
Door Sensor
        │
        ▼
Simulator
        │
        ▼
Mosquitto MQTT Broker
        │
        ▼
Inference Service
        │
        ├────────► SQLite Alert Database
        │
        ├────────► Live Feature Log
        │
        └────────► Drift Monitor
                        │
                        ▼
              FreightBridge Operations Centre
```

---

# MQTT Topics

Temperature

```
logibridge/trucks/TRUCK01/temperature
```

Vibration

```
logibridge/trucks/TRUCK01/vibration
```

Door

```
logibridge/trucks/TRUCK01/door
```

Alerts

```
logibridge/operations/TRUCK01/alerts
```

---

# Publish–Subscribe Flow

1. The simulator continuously generates sensor measurements.

2. Sensor values are published to the Mosquitto MQTT broker.

3. The inference service subscribes to all required sensor topics.

4. Sliding windows are maintained for feature extraction.

5. Extracted features are normalized before inference.

6. The TensorFlow Lite model predicts the truck operating state.

7. Warning and Critical events are written to the local SQLite database.

8. Once connectivity becomes available, alerts are synchronized with the FreightBridge Operations Centre.

---

# Advantages of MQTT

- Lightweight protocol
- Low bandwidth usage
- Low latency
- Reliable publish-subscribe communication
- Loose coupling between software components
- Suitable for IoT and Edge AI applications

---

# Offline Operation

During network outages:

- MQTT communication continues locally.
- TensorFlow Lite inference continues without interruption.
- Alerts are stored in the SQLite database.
- Feature logging continues.
- Synchronization resumes automatically after connectivity is restored.

This architecture ensures uninterrupted monitoring throughout transportation.