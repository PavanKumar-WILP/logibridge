FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cacheable)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY data_pipeline/ ./data_pipeline/
COPY inference/ ./inference/
COPY monitoring/ ./monitoring/

# Copy model LAST for OTA layer caching
COPY inference/model.tflite ./inference/model.tflite

# Runtime configuration
ENV MQTT_BROKER=host.docker.internal
ENV MQTT_PORT=1883
ENV TRUCK_ID=TRUCK01
ENV MODEL_PATH=/app/inference/model.tflite

CMD ["python", "-u", "inference/inference_service.py"]