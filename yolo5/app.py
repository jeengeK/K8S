import boto3
import os
import time
import logging
import requests
import numpy as np
import torch
from pymongo import MongoClient, errors
from PIL import Image
import io
import json
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load AWS credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "eu-north-1")
sqs_queue_url = os.getenv("SQS_QUEUE_URL")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
polybot_url = os.getenv("POLYBOT_URL", "http://svc-polybot:8443/6")

logging.info(f"Value of MONGO_CONNECTION_STRING from environment: '{mongo_connection_string}'")

# Initialize AWS clients
sqs = boto3.client(
    'sqs',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

s3 = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# MongoDB connection
if not mongo_connection_string:
    logging.error("MONGO_CONNECTION_STRING environment variable not set.")
    exit(1)

try:
    mongo_client = MongoClient(mongo_connection_string)
    db = mongo_client['yolo5_db']
    predictions_collection = db['predictions']
    logging.info("Connected to MongoDB successfully.")
except errors.PyMongoError as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    exit(1)

# Load YOLOv5 model
model = None
try:
    model_path = 'yolov5'
    model = torch.hub.load(model_path, 'custom', path='yolov5s.pt', source='local')
    logging.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLOv5 model: {e}")
    model = None

def process_sqs_message(sqs_client): # added sqs_client
    """Poll SQS queue and process messages."""
    while True:
        if not sqs_queue_url:
            logging.warning("SQS_QUEUE_URL not set. Skipping SQS polling.")
            time.sleep(10)
            continue

        logging.info("Polling SQS for messages...")

        response = sqs_client.receive_message( # using sqs_client
            QueueUrl=sqs_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )

        if 'Messages' in response:
            logging.info("Messages received from SQS.")
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            try:
                job_data = json.loads(message['Body'])
            except json.JSONDecodeError:
                logging.error(f"Error decoding SQS message body (not JSON): {message['Body']}")
                sqs_client.delete_message( # using sqs_client
                    QueueUrl=sqs_queue_url,
                    ReceiptHandle=receipt_handle
                )
                continue

            img_name = job_data['imgName']
            chat_id = job_data['chat_id']

            logging.info(f"Processing job for {img_name}")

            try:
                response = s3.get_object(Bucket=s3_bucket_name, Key=img_name)
                image_data = response['Body'].read()

                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image = np.array(image)

                if model is not None:
                    results = model(image)
                    predictions = [
                        {
                            "class": int(pred[5]),
                            "label": model.names[int(pred[5])],
                            "confidence": float(pred[4]),
                            "bbox": [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]
                        }
                        for pred in results.xyxy[0].tolist()
                    ]

                    prediction_id = str(predictions_collection.insert_one({
                        "imgName": img_name,
                        "chat_id": chat_id,
                        "predictions": predictions,
                        "timestamp": time.time()
                    }).inserted_id)
                    logging.info(f"Prediction results saved with predictionId: {prediction_id}")

                    send_results_to_polybot(prediction_id, chat_id)
                else:
                    logging.error("Model is not loaded")
            except Exception as e:
                logging.error(f"Error processing job {img_name}: {e}")

            sqs_client.delete_message( # using sqs_client
                QueueUrl=sqs_queue_url,
                ReceiptHandle=receipt_handle
            )
        else:
            logging.info("No messages in the queue. Waiting...")

def send_results_to_polybot(prediction_id, chat_id):
    """Send the processed results to Polybot's /results endpoint."""
    try:
        response = requests.post(polybot_url, json={"predictionId": prediction_id, "chat_id": chat_id})
        if response.status_code == 200:
            logging.info(f"Results sent to Polybot for chat_id {chat_id}")
        else:
            logging.error(f"Failed to send results to Polybot for chat_id {chat_id}, status code: {response.status_code}, response: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending results to Polybot: {e}")

if __name__ == "__main__":
    process_sqs_message(sqs) # passing sqs