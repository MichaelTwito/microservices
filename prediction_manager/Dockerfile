FROM python:3.9-slim-buster

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY . /opt/app
WORKDIR /opt/app

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
CMD [ "python", "grpc_server.py"]

EXPOSE 50051