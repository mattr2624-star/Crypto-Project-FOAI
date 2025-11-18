import json
from kafka import KafkaConsumer
import mlflow

# --- Configuration ---
KAFKA_BROKER = "cp-kafka-1:9092"  # Use the Docker service name
KAFKA_TOPIC = "test_topic"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # MLflow server inside container
EXPERIMENT_NAME = "Kafka_Logging_Experiment"

# --- Set up MLflow ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("Starting Kafka consumer and MLflow logger...")

# --- Set up Kafka consumer ---
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset='earliest',  # Start from beginning if no offsets
    enable_auto_commit=True,
    group_id='mlflow_logger_group',
    value_deserializer=lambda x: x.decode('utf-8')
)

# --- Consume messages and log to MLflow ---
for message in consumer:
    msg_value = message.value
    print(f"Received message: {msg_value}")

    # Start an MLflow run to log the message
    with mlflow.start_run():
        mlflow.log_text(msg_value, artifact_file=f"message_{message.offset}.txt")
        print(f"Logged message offset {message.offset} to MLflow.")
