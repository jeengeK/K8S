import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
from botocore.exceptions import NoCredentialsError
import json
import requests  # Import the requests library
import uuid  # Import uuid for generating unique IDs






try:
  BUCKET_NAME = os.environ.get('bucket_name2')
  SQS_URL = os.environ.get('SQS_URL')
  TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
  POLYBOT_RESULTS_URL = os.environ.get('POLYBOT_RESULTS_URL')  # URL to call back to PolyBot with results




  if not BUCKET_NAME:
      raise ValueError("BUCKET_NAME environment variable not set")
  if not SQS_URL:
      raise ValueError("SQS_URL environment variable not set")
  if not TELEGRAM_TOKEN:
      raise ValueError("TELEGRAM_TOKEN environment variable not set")
  if not POLYBOT_RESULTS_URL:
      raise ValueError("POLYBOT_RESULTS_URL environment variable not set")








except ValueError as e:
  raise RuntimeError(f"Missing or empty required environment variable: {e}")
except KeyError as e:
  raise RuntimeError(f"Missing required environment variable: {e}")








class Bot:




  def __init__(self, token):
      # create a new instance of the TeleBot class.
      # all communication with Telegram servers are done using self.telegram_bot_client
      self.telegram_bot_client = telebot.TeleBot(token)
      logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')




  def send_text(self, chat_id, text):
      self.telegram_bot_client.send_message(chat_id, text)




  def send_text_with_quote(self, chat_id, text, quoted_msg_id):
      self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)




  def is_current_msg_photo(self, msg):
      if isinstance(msg, dict): # checking the type of message is dict
          return 'photo' in msg
      elif isinstance(msg, telebot.types.Message): # checking the type of message is telebot.types.Message
          return msg.photo is not None
      return False




  def download_user_photo(self, msg):
      """
      Downloads the photos that sent to the Bot to `photos` directory (should be existed)
      :return:
      """
      if not self.is_current_msg_photo(msg):
          raise RuntimeError(f'Message content of type \'photo\' expected')




      if isinstance(msg, dict): # if message is dict then access photo data in dict way
          file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
      elif isinstance(msg, telebot.types.Message): # if message is telebot.types.Message then access photo data in telebot.types.Message way
          file_info = self.telegram_bot_client.get_file(msg.photo[-1].file_id)
      else:
          raise TypeError(f"Unexpected message type: {type(msg)}")




      data = self.telegram_bot_client.download_file(file_info.file_path)
      folder_name = file_info.file_path.split('/')[0]




      if not os.path.exists(folder_name):
          os.makedirs(folder_name)




      with open(file_info.file_path, 'wb') as photo:
          photo.write(data)




      return file_info.file_path




  def send_photo(self, chat_id, img_path):
      if not os.path.exists(img_path):
          raise RuntimeError("Image path doesn't exist")




      self.telegram_bot_client.send_photo(
          chat_id,
          InputFile(img_path)
      )








  def handle_photo_message(self, msg):
      chat_id = None
      if isinstance(msg, dict): # checking the type of message is dict
          chat_id = msg['chat']['id']
      elif isinstance(msg, telebot.types.Message): # checking the type of message is telebot.types.Message
          chat_id = msg.chat.id




      else:
          raise TypeError(f"Unexpected message type: {type(msg)}")








      photo_path = self.download_user_photo(msg)




      # Generate a unique prediction ID
      prediction_id = str(uuid.uuid4())




      # upload the image to S3 Bucket
      s3 = boto3.client('s3')
      bucket_name = BUCKET_NAME
      # Include prediction_id in the S3 key
      s3_image_key_upload = f'{chat_id}_{prediction_id}_teleBOT_picture.jpg'
      try:
          # Upload predicted image back to S3
          s3.upload_file(str(photo_path), bucket_name, s3_image_key_upload)
          logger.info(f"File uploaded successfully to {bucket_name}/{s3_image_key_upload}")
      except FileNotFoundError:
          logger.error("The file was not found.")
          self.send_text(chat_id, "Error: Original image not found.")
          return
      except NoCredentialsError:
          logger.error("AWS credentials not available.")
          self.send_text(chat_id, "Error: AWS credentials not available.")
          return
      except Exception as e:
          logger.error(f"Error uploading file: {e}")
          self.send_text(chat_id, f"Error uploading file: {e}")
          return








      # send an HTTP request to the `SQS` service for prediction
      params = {
          "imgName": s3_image_key_upload,
          "predictionId": prediction_id,  # Include predictionId in the SQS message
          "chat_id": chat_id
      }
      sqs_client = boto3.client('sqs', region_name='eu-north-1')
      try:
          response = sqs_client.send_message(
              QueueUrl=str(SQS_URL),
              MessageBody=json.dumps(params)
          )
          logger.info(f"Message sent to SQS. Message ID: {response['MessageId']}")
          self.send_text(chat_id, "Image sent for processing.  Please wait...") # Immediate feedback


      except Exception as e:
          logger.error(f"Error sending message to SQS: {e}")
          self.send_text(chat_id, f"Error sending message to SQS: {e}")
          return

      # Removed download and send image logic.  This will be handled by Polybot callback

  def handle_message(self, msg):
      """Bot Main message handler"""
      logger.info(f'Incoming message: {msg}')
      chat_id = None
      if isinstance(msg, dict): # checking the type of message is dict
          if 'text' in msg:
              self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')
          elif self.is_current_msg_photo(msg):
              self.handle_photo_message(msg)
          else:
              self.send_text(msg['chat']['id'], "Unsupported message type")




      elif isinstance(msg, telebot.types.Message): # checking the type of message is telebot.types.Message
          if msg.text:
              self.send_text(msg.chat.id, f'Your original message: {msg.text}')
          elif msg.photo:
              self.handle_photo_message(msg)
          else:
              self.send_text(msg.chat.id, "Unsupported message type")




      else:
          self.send_text(chat_id, "Unsupported message type")








class ObjectDetectionBot(Bot):
  def __init__(self):
      super().__init__(TELEGRAM_TOKEN)




  def handle_message(self, msg):
      logger.info(f'Incoming message: {msg}')








      if self.is_current_msg_photo(msg):
          self.handle_photo_message(msg)
      elif isinstance(msg, dict) and 'text' in msg:
          self.send_text(msg['chat']['id'], f"this is your message: {msg['text']}")
      elif isinstance(msg, telebot.types.Message) and msg.text:
          self.send_text(msg.chat.id, f"this is your message: {msg.text}")
      else:
          chat_id = None
          if isinstance(msg, dict):
               chat_id = msg.get('chat', {}).get('id')
          elif isinstance(msg, telebot.types.Message):
              chat_id = msg.chat.id
          self.send_text(chat_id, "Unsupported message type")



# Main function to run the bot
def main():
  bot = ObjectDetectionBot()




  # Delete the webhook
  try:
      response = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook")
      response.raise_for_status()  # Raise an exception for bad status codes
      logger.info(f"Webhook deleted: {response.json()}")
  except requests.exceptions.RequestException as e:
      logger.error(f"Error deleting webhook: {e}")
      # Handle the error appropriately (e.g., exit the program, retry later)




  # Define the message handler
  @bot.telegram_bot_client.message_handler(func=lambda message: True, content_types=['text', 'photo'])
  def echo_all(message):
      bot.handle_message(message)  # Pass the message object directly




  # Start the bot
  logger.info("Bot started.  Listening for messages...")
  bot.telegram_bot_client.infinity_polling()



if __name__ == "__main__":
  main()



