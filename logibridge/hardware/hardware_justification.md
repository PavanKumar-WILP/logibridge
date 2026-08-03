# Hardware Justification

## Overview

The LogiBridge Edge AI system is designed to execute directly on resource-constrained edge devices installed inside refrigerated transport vehicles.

Unlike cloud computing platforms, edge hardware provides limited CPU, memory and storage resources. The selected software architecture therefore prioritises lightweight processing, low latency and efficient resource utilisation.

---

# Selected Hardware

The proposed deployment platform is Raspberry Pi 4 Model B.

Typical specifications include:

- Quad-core ARM Cortex-A72 CPU
- 4 GB RAM
- microSD storage
- Ethernet
- Wi-Fi
- USB interfaces
- GPIO support

These specifications are sufficient for continuous MQTT communication, TensorFlow Lite inference and local data storage.

---

# Why Raspberry Pi?

The Raspberry Pi offers several advantages for edge deployments.

## Low Power Consumption

Continuous operation inside transport vehicles requires energy-efficient hardware.

The Raspberry Pi consumes significantly less power than conventional desktop systems.

---

## Sufficient Processing Capability

TensorFlow Lite models require only modest computational resources.

The trained MLP classifier contains relatively few parameters, allowing inference to execute within milliseconds.

---

## Local Storage

SQLite databases store alerts locally whenever cellular communication becomes unavailable.

This enables reliable offline operation.

---

## Networking Support

Built-in Ethernet and Wi-Fi simplify communication with:

- MQTT Broker
- Operations Centre
- Local monitoring tools

---

## Expandability

Additional sensors can easily be integrated using GPIO interfaces.

Future extensions may include:

- Humidity sensors
- GPS modules
- CO₂ sensors
- Accelerometers

without major architectural changes.

---

# Why TensorFlow Lite?

TensorFlow Lite is specifically designed for embedded devices.

Advantages include:

- Small model size
- Reduced memory usage
- Fast inference
- Hardware optimisation
- INT8 quantization support

These characteristics make TensorFlow Lite suitable for Edge AI deployment.

---

# Storage Requirements

The complete application requires relatively small storage.

Approximate components include:

- Source code
- TensorFlow Lite model
- SQLite database
- Benchmark results
- Monitoring logs

The total storage requirement remains well within the capabilities of a Raspberry Pi.

---

# Conclusion

The selected hardware platform provides an appropriate balance between computational performance, power efficiency, cost and deployment flexibility, making it well suited for real-time Edge AI cold-chain monitoring.