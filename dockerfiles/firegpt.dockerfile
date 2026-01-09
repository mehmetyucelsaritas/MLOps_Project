FROM python:3.12-slim AS base
LABEL authors="mehmet"

LABEL team="group-x"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./src/mlops_project ./src/mlops_project
COPY ./models/ ./models/
COPY ./data/processed/chunking_metadata.json ./data/processed/chunking_metadata.json
COPY pyproject.toml ./

RUN pip install -e .

RUN [ -f ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ] || hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models
# RUN [ -f ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf ] || hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./models
# RUN [ -f ./models/Phi-3-mini-4k-instruct-q4.gguf ] || hf download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models

EXPOSE $PORT
ENTRYPOINT ["streamlit", "run", "src/mlops_project/gui/gui.py", "--server.port=${PORT}", "--server.address=0.0.0.0", "--server.headless=true"]
