#FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM python:3.12 AS base
LABEL authors="Wbtqu"

LABEL team="AMI_Group_1"
LABEL version="Presentation State"

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

#RUN wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run \
#RUN sudo sh cuda_12.9.1_575.57.08_linux.run
#RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python

RUN pip install llama-cpp-python
EXPOSE 8501
WORKDIR ./
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./out ./out
COPY ./RAG_LLM ./RAG_LLM
COPY ./src ./src
COPY ./GUI ./GUI
COPY ./models/ ./models/
COPY .env* ./

RUN [ -f ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ] || hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models
# RUN [ -f ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf ] || huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./models
# RUN [ -f ./models/Phi-3-mini-4k-instruct-q4.gguf ] || huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models

#CMD ["ls", "models"]
ENTRYPOINT ["streamlit", "run", "GUI/GUI.py"]
