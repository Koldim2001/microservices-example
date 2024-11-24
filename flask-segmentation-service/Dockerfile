FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Сначала ставим torch c gpu
RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY ./flask-segmentation-service/requirements.txt /app
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
COPY /flask-segmentation-service /app

EXPOSE 5050

CMD python3 flask_api_seg.py