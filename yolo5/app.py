import json
import logging
import os
import time

import boto3
import pymongo
import torch
import requests

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables, with hardcoded defaults for this example only.
# NEVER DO THIS IN REAL PRODUCTION CODE.  Use a more secure method.
# The correct values must be set as environment variables, these are just fallback defaults.
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "netflix.jeenge")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "https://sqs.eu-north-1.amazonaws.com/352708296901/itsik-netflix-events")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "AKIAVEHYNQDC5MO5GQK7")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "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")
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7479409954:AAEfM0yr67JzXeqw2RsCpFPcWtHq6jv-pec")
TELEGRAM_APP_URL = os.environ.get("TELEGRAM_APP_URL", "https://t.me/itsikINT2024_bot")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")  # Add default for MongoDB
POLYBOT_RESULTS_URL = os.environ.get("POLYBOT_RESULTS_URL", "http://localhost:8443/results") # Add default for POLYBOT_RESULTS_URL
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
DELETE_INVALID = os.environ.get("DELETE_INVALID", "False").lower() == "true"


def connect_to_mongodb():
    """Connects to MongoDB."""
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        client.admin.command('ping')  # Test the connection
        logging.info("MongoDB connection successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        logging.error(f"MongoDB connection failed: {e}")
        return None


def load_yolov5_model():
    """Loads the YOLOv5 model."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can change 'yolov5s' to other versions
        logging.info("Successfully loaded YOLOv5 model")
        return model
    except Exception as e:
        logging.error(f"Failed to load YOLOv5 model: {e}")
        return None


def connect_to_sqs():
    """Connects to SQS."""
    try:
        # Use AWS credentials from environment variables
        sqs = boto3.client('sqs', region_name=AWS_REGION,
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        logging.info("SQS connection successful")
        return sqs
    except Exception as e:
        logging.error(f"SQS connection failed: {e}")
        return None


def download_image_from_s3(s3_client, bucket_name, image_name):
    """Downloads an image from S3 to a temporary file."""
    try:
        # Create a temporary file
        temp_file = f"/tmp/{image_name}"  # Use /tmp directory which is writable

        # Download the file
        s3_client.download_file(bucket_name, image_name, temp_file)
        logging.info(f"Downloaded image {image_name} from S3 to {temp_file}")
        return temp_file
    except Exception as e:
        logging.error(f"Error downloading image {image_name} from S3: {e}")
        return None


def process_message(message, model, mongodb_client, sqs_client, queue_url):
    """Processes a single message from SQS."""
    try:
        body = message['Body']
        if body.startswith('{') or body.startswith('['):  # check if looks like json.
            try:
                data = json.loads(body)
                logging.info(f"Processing message: {data}")
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON received: {body} - Error: {e} (ReceiptHandle: {message['ReceiptHandle']})")
                if DELETE_INVALID:
                    delete_message(sqs_client, queue_url, message['ReceiptHandle'])
                    logging.warning(f"Deleted invalid message {message['ReceiptHandle']}")
                else:
                    logging.warning(f"Message {message['ReceiptHandle']} processing failed, will retry unless deleted")
                return
        else:
            logging.error(f"Received non-JSON message: {body} (ReceiptHandle: {message['ReceiptHandle']})")
            if DELETE_INVALID:
                delete_message(sqs_client, queue_url, message['ReceiptHandle'])
                logging.warning(f"Deleted non-JSON message {message['ReceiptHandle']}")
            else:
                logging.warning(f"Message {message['ReceiptHandle']} processing failed, will retry unless deleted")
            return

        image_name = data.get('imgName')  # Changed to imgName (S3 key)
        prediction_id = data.get('predictionId')
        chat_id = data.get('chat_id')

        if not image_name or not prediction_id or not chat_id:
            logging.error(f"Missing image_url, predictionId, or chat_id in message: {data}")
            return

        try:
            # Download the image from S3
            s3_resource = boto3.resource('s3', region_name=AWS_REGION,
                                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            temp_image_path = download_image_from_s3(s3_resource.meta.client, S3_BUCKET_NAME, image_name)

            if not temp_image_path:
                logging.error(f"Failed to download image {image_name} from S3.  Aborting message processing.")
                return

            # Run inference
            model_results = model(temp_image_path)  # Pass the local path to YOLOv5
            model_results.print()  # print results to log, for debugging, remove in production.

            # Process the results and create a prediction summary
            detections = []
            for *xyxy, conf, cls in model_results.xyxy[0]:  # Iterate detections
                label = model_results.names[int(cls)]
                confidence = float(conf)
                x1, y1, x2, y2 = map(float, xyxy)
                detections.append({
                    'class': label,
                    'confidence': confidence,
                    'box': [x1, y1, x2, y2]
                })

            prediction_summary = {
                "prediction_id": prediction_id,
                "chat_id": int(chat_id),  # Ensure chat_id is an integer
                "detections": detections,
                "processing_time": time.time()  # Add processing time
            }

            # Save the results to MongoDB
            try:
                db = mongodb_client["polybot-info"]  # Access Database
                collection = db["prediction_images"]  # Access Collection
                document = {"prediction_summary": prediction_summary}
                collection.insert_one(document)
                logging.info(f"Successfully saved results to MongoDB for prediction ID: {prediction_id}")

            except pymongo.errors.PyMongoError as e:
                logging.error(f"Error saving to MongoDB: {e}")
                return  # Stop processing if saving to MongoDB fails

            # Call the PolyBot results endpoint
            try:
                response = requests.post(f"{POLYBOT_RESULTS_URL}?predictionId={prediction_id}")
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                logging.info(
                    f"Successfully called PolyBot results endpoint for prediction ID: {prediction_id}. Status Code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error calling PolyBot results endpoint: {e}")

            logging.info(f"Successfully processed image {image_name}")

            # Clean up the temporary image file
            try:
                os.remove(temp_image_path)
                logging.info(f"Deleted temporary file {temp_image_path}")
            except OSError as e:
                logging.warning(f"Error deleting temporary file {temp_image_path}: {e}")

        except Exception as image_processing_error:
            logging.error(f"Error processing image {image_name}: {image_processing_error}")
            return  # stop processing if image processing failed.

        # Delete the message from the queue after successful processing
        delete_message(sqs_client, queue_url, message['ReceiptHandle'])
        logging.info(f"Deleted message {message['ReceiptHandle']} from SQS")

    except Exception as e:
        logging.exception(f"Error processing message {message.get('MessageId', 'Unknown')}: {e}")
        # Optionally, you might want to requeue or handle the error differently here.
        # If you don't delete the message, it will automatically become visible again after the VisibilityTimeout.
        logging.warning(f"Message processing failed, will retry unless deleted")


def delete_message(sqs_client, queue_url, receipt_handle):
    """Deletes a message from the SQS queue."""
    try:
        sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
    except Exception as e:
        logging.error(f"Failed to delete message {receipt_handle}: {e}")


def main():
    """Main function to start the YOLOv5 microservice."""

    logging.info(f"Starting YOLOv5 microservice on Python {os.sys.version} ({os.sys.platform})")
    logging.info(f"Config: MongoDB={MONGODB_URI}, SQS={SQS_QUEUE_URL}, Interval={POLL_INTERVAL}s, DeleteInvalid={DELETE_INVALID}, BUCKET={S3_BUCKET_NAME}, POLYBOT={POLYBOT_RESULTS_URL}, REGION={AWS_REGION}")

    # Validate that mandatory configuration is present
    if not all([MONGODB_URI, SQS_QUEUE_URL, S3_BUCKET_NAME, POLYBOT_RESULTS_URL, AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY, AWS_REGION]):
        logging.critical("Missing mandatory configuration. Exiting.")
        return

    mongodb_client = connect_to_mongodb()
    if not mongodb_client:
        logging.critical("Failed to connect to MongoDB. Exiting.")
        return

    model = load_yolov5_model()
    if not model:
        logging.critical("Failed to load YOLOv5 model. Exiting.")
        return

    sqs_client = connect_to_sqs()
    if not sqs_client:
        logging.critical("Failed to connect to SQS. Exiting.")
        return

    queue_url = SQS_QUEUE_URL

    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,  # Adjust as needed
                WaitTimeSeconds=5,  # Long polling to reduce empty responses
            )

            messages = response.get('Messages', [])
            logging.info(f"Received {len(messages)} messages from SQS queue")

            for message in messages:
                logging.info(f"Raw SQS message: {message.get('Body', 'No Body')}")  # Log the raw message
                process_message(message, model, mongodb_client, sqs_client, queue_url)

        except Exception as e:
            logging.exception(f"Error receiving or processing messages: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()