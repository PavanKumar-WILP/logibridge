FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data_pipeline/ ./data_pipeline/
COPY monitoring/ ./monitoring/


COPY inference/inference_service.py ./inference/


COPY inference/model.tflite ./inference/model.tflite

ENV MQTT_BROKER=host.docker.internal \
    MQTT_PORT=1883 \
    TRUCK_ID=TRUCK01 \
    MODEL_PATH=/app/inference/model.tflite

CMD ["python", "-u", "inference/inference_service.py"]