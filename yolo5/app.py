import boto3
import os
import time
import logging
import requests
import numpy as np
import torch
from pymongo import MongoClient
import pymongo
from PIL import Image
import io
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
mongo_uri = os.getenv("MONGO_URI")
polybot_url = os.getenv("POLYBOT_URL", f'http://svc-polybot:8443/results')

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

import boto3

# Initialize SQS client
sqs = boto3.client(
    'sqs',
    region_name='eu-north-1',  # Ensure the correct region
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Queue URL
queue_url = 'https://sqs.eu-north-1.amazonaws.com/352708296901/itsik-netflix-events'

# Receive messages from the queue
try:
    response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)

    if 'Messages' in response:
        print(f"Received message: {response['Messages'][0]['Body']}")
    else:
        print("No messages in the queue.")
except sqs.exceptions.QueueDoesNotExist:
    print("Queue does not exist!")
except Exception as e:
    print(f"An error occurred: {e}")

# MongoDB connection
mongo_client = MongoClient('mongodb://mongodb:27017')
db = mongo_client['yolo5_db']
predictions_collection = db['predictions']

# Load YOLOv5 model
try:
    # Assuming your yolov5 directory is in the same directory as app.py
    model_path = 'yolov5'
    model = torch.hub.load(model_path, 'custom', path='yolov5s.pt', source='local')
    logging.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLOv5 model: {e}")
    model = None


def process_sqs_message():
    """Poll SQS queue and process messages."""
    while True:
        response = sqs.receive_message(
            QueueUrl=sqs_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10  # Long poll for 10 seconds
        )

        if 'Messages' in response:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            try:
                job_data = eval(message['Body'])  # Convert string back to dictionary
            except (SyntaxError, NameError):
                logging.error(f"Error decoding SQS message body: {message['Body']}")
                # Optionally delete the message if it can't be decoded
                sqs.delete_message(
                    QueueUrl=sqs_queue_url,
                    ReceiptHandle=receipt_handle
                )
                continue

            img_name = job_data['imgName']
            chat_id = job_data['chat_id']

            logging.info(f"Processing job for {img_name}")

            try:
                # Download image from S3
                response = s3.get_object(Bucket=s3_bucket_name, Key=img_name)
                image_data = response['Body'].read()

                # Convert image to NumPy array
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image = np.array(image)

                if model is not None:
                    results = model(image)  # Perform inference
                    predictions = [
                        {
                            "class": int(pred[5]),
                            "label": model.names[int(pred[5])],
                            "confidence": float(pred[4]),
                            "bbox": [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]
                        }
                        for pred in results.xyxy[0].tolist()
                    ]

                    # Save results to MongoDB
                    prediction_id = str(predictions_collection.insert_one({
                        "imgName": img_name,
                        "chat_id": chat_id,
                        "predictions": predictions,
                        "timestamp": time.time()
                    }).inserted_id)
                    logging.info(f"Prediction results saved with predictionId: {prediction_id}")

                    # Send result to Polybot
                    send_results_to_polybot(prediction_id, chat_id)
                else:
                    logging.error("Model is not loaded")
            except Exception as e:
                logging.error(f"Error processing job {img_name}: {e}")

            # Delete the message from the queue
            sqs.delete_message(
                QueueUrl=sqs_queue_url,
                ReceiptHandle=receipt_handle
            )
        else:
            logging.info("No messages in the queue. Waiting...")


import requests
import logging

def send_results_to_polybot(prediction_id, chat_id):
    """Send the processed results to Polybot's /results endpoint."""
    polybot_url = f'http://svc-polybot:8443/results'
    payload = {"predictionId": prediction_id, "chat_id": chat_id}
    try:
        response = requests.post(polybot_url, json=payload)
        if response.status_code == 200:
            logging.info(f"Results sent to Polybot for chat_id {chat_id}")
        else:
            logging.error(f"Failed to send results to Polybot for chat_id {chat_id}, status code: {response.status_code}, response: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending results to Polybot: {e}")


if __name__ == "__main__":
    process_sqs_message()