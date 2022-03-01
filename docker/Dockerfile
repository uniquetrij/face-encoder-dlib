FROM python:3.9
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt install -y git screen nano curl wget
RUN mkdir /app && chmod -R 777 /app
RUN apt install -y cmake libgl1
COPY face_auth app/face_auth
COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
COPY script.sh app/script.sh
WORKDIR /app
ENV PYTHONPATH /app
CMD bash script.sh