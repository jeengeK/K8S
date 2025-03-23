import boto3
import os
import time
import logging
from PIL import Image
import io
import torch
import pymongo
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load AWS credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")
sqs_queue_url = os.getenv("SQS_QUEUE_URL")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
mongo_uri = os.getenv("MONGO_URI")

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

# MongoDB client setup
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["yolo5_db"]
predictions_collection = db["predictions"]

# Load YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logging.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLOv5 model: {e}")
    model = None

def process_sqs_message():
    """Poll SQS queue and process messages."""
    while True:
        # Receive message from SQS
        response = sqs.receive_message(
            QueueUrl=sqs_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10  # Long poll for 10 seconds
        )

        if 'Messages' in response:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            job_data = eval(message['Body'])  # Convert string back to dictionary

            img_name = job_data['imgName']
            chat_id = job_data['chat_id']

            logging.info(f"Processing job for {img_name}")

            # Process the image
            try:
                response = s3.get_object(Bucket=s3_bucket_name, Key=img_name)
                image_data = response['Body'].read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')

                if model is not None:
                    results = model(image)
                    predictions = results.pandas().xyxy[0].to_dict(orient="records")

                    # Save results to MongoDB
                    prediction_id = predictions_collection.insert_one({
                        "imgName": img_name,
                        "chat_id": chat_id,
                        "predictions": predictions,
                        "timestamp": time.time()
                    }).inserted_id
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

def send_results_to_polybot(prediction_id, chat_id):
    """Send the processed results to Polybot's /results endpoint."""
    import requests

    polybot_url = os.getenv("POLYBOT_URL", "http://polybot_service/results")
    try:
        response = requests.post(f"{polybot_url}?predictionId={prediction_id}")
        if response.status_code == 200:
            logging.info(f"Results sent to Polybot for chat_id {chat_id}")
        else:
            logging.error(f"Failed to send results to Polybot for chat_id {chat_id}")
    except Exception as e:
        logging.error(f"Error sending results to Polybot: {e}")

if __name__ == "__main__":
    process_sqs_message()
