FROM python:3.9-slim-buster

  WORKDIR /app

  RUN pip uninstall openvino openvino-dev -y
  RUN apt-get update && apt-get install -y libgl1-mesa-glx
  RUN apt-get update && apt-get install -y libglib2.0-0




  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  COPY . .

  CMD ["python", "app.py"]