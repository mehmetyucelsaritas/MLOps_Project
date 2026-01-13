
# FireGPT: Intelligent Wildfire Response Assistant Project Description

## 1. Overall Goal of the Project

The primary goal of this project is to design and implement **FireGPT**, an intelligent, deployable decision support system intended to assist frontline wildfire responders in high-stakes emergency scenarios. Motivated by the increasing frequency and severity of wildfires due to climate change and human activities, the system addresses the urgent need for data-driven management in dynamic fire conditions.

FireGPT integrates **Retrieval-Augmented Generation (RAG)** with a **Large Language Model (LLM)** to provide context-aware action plans. By prioritizing operational robustness and low-latency performance, FireGPT aims to bridge advanced AI techniques with practical field constraints to improve safety and response effectiveness.

## 2. Data Strategy (Initial & Pre-processing)

The system runs on a high-quality knowledge base constructed from curated documents sourced from global firefighting and emergency response agencies.

### Initial Data Sources

The initial corpus consists of approximately **20 documents**, selected for their breadth and relevance to wildfire management. The data covers various geographic regions and operational doctrines, including:

- **United States:** The *Aerial Firefighting Use and Effectiveness (AFUE) Final Report* and case studies on fire behavior such as the South Canyon Fire.
- **Germany:** The *Waldbrandbericht 2022* and technical recommendations for aerial firefighting.
- **Australia:** The *National Aerial Firefighting Strategy 2021–26* and technical operating procedures from New South Wales.
- **Other Regions:** Strategic frameworks and guidance from Sweden, Portugal, and Canada.

### Data Pre-processing

To prepare this data for the RAG pipeline, the following steps are taken:

- **Image-to-Text Transcription:** Since many wildfire documents contain critical maps and diagrams, a custom Python script uses the Google Gemini API to transcribe these images into semantic text descriptions, which are inserted back into the documents.
- **Chunking:** Text is segmented into semantically coherent units of approximately 100 to 300 words to balance granularity with computational efficiency.
- **Metadata Annotation:** Chunks are annotated with source, region, and language to enable filtering during retrieval.

## 3. Models and Technical Architecture

The system is designed for both local and remote execution on resource-constrained hardware using models in the **GGUF format** via `llama.cpp`.

### Language Models (LLMs)

The project utilizes a modular orchestration layer that supports dynamic switching between the following models:

- **Mistral-7B Instruct:** Used for generating the most detailed and structured responses, capable of precise GPS references and multi-location planning.
- **Phi-3 Mini:** Intended as the primary balanced option; it offers strong reasoning capabilities and faster inference speeds compared to Mistral-7B, making it suitable for lightweight deployment.
- **TinyLlama:** Included as a fallback model for highly constrained environments due to its low resource footprint and fast response time.

### Embedding and Retrieval

- **Embedding Model:** **all-MiniLM-L6-v2** (Sentence Transformers) is used to encode text chunks into high-dimensional semantic vectors.
- **Vector Store:** **FAISS** (Facebook AI Similarity Search) is employed to store vectors and perform efficient approximate nearest neighbor searches.

# 🔥 FireGPT — Wildfire Decision Support with LLMs in Action

![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)
![Models](https://img.shields.io/badge/LLM%20Size-1.1B–7B-orange)
![Status](https://img.shields.io/badge/Service-Live-success)

## Quick Start

You can try FireGPT **without installation** using the live web demo:

> **Best for:** First-time users, quick testing

| Feature | Details |
|---------|--------|
| Initial Load Time | 5–15 seconds |
| Query Response Time | 3–5 minutes |
| Availability | Until **29/03/2026** |

For an example quary, see the [Query section](#query) below.

[![Open FireGPT Live Demo](https://img.shields.io/badge/Try%20Live-FireGPT-red?style=for-the-badge)](https://firegpt-image-361370435150.europe-west4.run.app)

Alternatively, If you want to run it faster, It takes 30-50 second to respond on local machine (M4 macbook air), I provide a dockerfile to build the app and get its dependencies with less headache. First, If you do not have you docker already installed visit [official docker page to install docker.](https://www.docker.com/get-started/)

Clone the repository:

```
git clone git@github.com:mehmetyucelsaritas/MLOps_Project.git
```
Even though the LLM models are automatically downloaded in the docker image, we recommend to download the models prior, as long recompilation caused by potential build issues can be avoided this way:

```
pip install huggingface-hub

hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./models
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models
hf download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models
```

Next, execute this line in a terminal (PowerShell) in the repositories root:

```
docker build -f dockerfiles/firegpt.dockerfile . -t firegpt:latest ; docker run -p 8080:8080 -e PORT=8080 -it firegpt
```

You can also use project scripts on terminal for processing data, building metadata, running firegpt and runnig firegpt gui respectively.

```
data
build_metadata
python3 src/mlops_project/main.py
streamlit run src/mlops_project/gui/gui.py
```

In the terminal you should see the link to the web interface.
Otherwise go to http://localhost:8501. For testing you can use following synthetic query (generated with ChatGPT, based on the Eaton Fire):

### Query
I’m facing a wildfire that started at coordinate (2.0,1.5) in the northwest quadrant of our operational map (grid origin at top-left (0,0), units in km). The fire has spread southeast, roughly following a ridge from (2.5,2.0) to (4.0,3.5), and it’s approaching the foothill community zone in the rectangular area between X =4.5–6.5km and Y=3.5–5.0km. There’s a natural drainage gulch along Y ≈1.0 where ember‑spotting is common, and a human-made access road/firebreak runs at X ≈3.0 from Y=0.5 to Y=5.0. Two evacuation shelters are located outside the fire area at approximately (8.5,6.5) and (9.0,7.0).

We have available assets:
• 1 Thermal IR drones
• 1 Drone with water carrying ability
• 2 trucks and crew

### Example result with mistral-7b:

    ACTION PLAN:

        Location: Fire Command Center (approximately (0.0, 0.0)) Action: Assemble the incident management team and assess the current situation using available resources and weather data. Required Resources: Thermal IR drones, weather data, operational map, communication equipment. Reasoning: Early assessment and understanding of the fire situation is crucial for effective response planning and resource allocation. Navigation Hint: The fire command center is located at the grid origin of the operational map.

        Location: Fire Perimeter near (4.0, 3.5) Action: Deploy the first firefighting crew to contain the fire and secure the human-made firebreak. Required Resources: 1 truck and crew. Reasoning: The fire is approaching the foothill community zone, and securing the firebreak is essential to prevent the fire from spreading further. Navigation Hint: Travel south along the Y-axis from the fire command center to reach the fire perimeter.

        Location: Fire Perimeter near (2.5, 2.0) Action: Deploy the second firefighting crew to engage the fire from the northwest and create a containment line. Required Resources: 1 truck and crew. Reasoning: The fire is following a ridge from this location, and containing it here will help prevent the fire from spreading further in that direction. Navigation Hint: Travel west along the X-axis from the fire command center to reach the fire perimeter.

        Location: Natural Drainage Gulch (approximately Y ≈ 1.0) Action: Deploy the thermal IR drone to monitor for ember spotting and identify potential hotspots. Required Resources: Thermal IR drone. Reasoning: The natural drainage gulch is a common area for ember spotting, and early detection and containment of these hotspots is crucial to prevent the fire from spreading. Navigation Hint: Travel south along the Y-axis from the fire command center to reach the natural drainage gulch.

        Location: Evacuation Shelter at (8.5, 6.5) Action: Deploy the drone with water carrying ability to support evacuation efforts and provide water to residents. Required Resources: Drone with water carrying ability. Reasoning: Evacuation shelters need water and support to accommodate residents, and the drone can help provide this essential resource. Navigation Hint: Travel northwest from the fire command center to reach the evacuation shelter.

    SOURCES USED:

        Vegetation Fire Fighting Training Manual for Fire Departments Version 1, Region: Germany
        Fire Management Today Volume 63, No. 4, Fall 2003, Region: International/United States/Australia/New Zealand
        The South Canyon Fire Behavior Report, Region: Colorado, United States

### Queries for evaluation
You can find test_queries.json under ./data/testset/ for evaluation. For your reference you can also find our evaluation results for each query with different models in ./outputs/evaluation_results/. M7 stands for Mistral-7B; P3 stands for Phi3 Mini; and TL stands for TinyLlama.
