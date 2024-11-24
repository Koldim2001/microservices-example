FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY ./flask-titanic-service/requirements.txt /app
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
COPY /flask-titanic-service /app

CMD python3 flask_api_titanic.py