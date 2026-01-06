import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from mlops_project.model.retrieve import Retrieve
from mlops_project.rag_llm.rag_runner import RAGPipeline
from mlops_project.utility import JsonHandler

# Model mapping
MODEL_PATHS = {
    "Phi-3": "./models/Phi-3-mini-4k-instruct-q4.gguf",
    "Mistral-7B": "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "TinyLlama": "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
}

st.title("FireGPT - Wildfire Emergency Assistant")

# Sidebar Model Selector
selected_model = st.sidebar.selectbox("Select LLM Model", list(MODEL_PATHS.keys()))
n_gpu_layers = st.sidebar.slider("GPU Layers", 0, 50, 0)
context_length = st.sidebar.slider("Context Length", 512, 4096, 4096, step=512)

# File uploader + map settings
uploaded_file = st.sidebar.file_uploader("Upload a map image (.png)")
st.sidebar.radio("Units", ["mi", "km", "m"], horizontal=True)
side1, side2 = st.sidebar.columns(2)
with side1:
    st.number_input("Image Height", 0.001, 250.0, value=10.0, step=0.001)
with side2:
    st.number_input("Image Width", 0.001, 250.0, value=20.0, step=0.001)

# Reset button
if st.sidebar.button("Start again"):
    st.session_state.query = ""
    st.rerun()


# Initialize query in session state
if "query" not in st.session_state:
    st.session_state.query = ""

# Weather & Time Inputs
col1, col2 = st.columns(2)
with col1:
    reqdate = st.date_input("Date of Fire Fighting Approach")
with col2:
    reqtime = st.time_input("Time of Fire Fighting Approach")

coli1, coli2, coli3, coli4, coli5 = st.columns(5)
with coli1:
    sky = st.selectbox("Weather", ["Clear Sky", "Cloudy", "Rain", "Snow/Hail"])
with coli2:
    temp = st.slider("Temperature (°C)", -20, 60, 20)
with coli3:
    humid = st.slider("Humidity (%)", 0, 100, 50)
with coli4:
    winddir = st.selectbox(
        "Wind Direction",
        ["No Wind", "North", "East", "South", "West", "North East", "North West", "South East", "South West"],
    )
with coli5:
    windspeed = st.slider("Wind Speed (m/s)", 0, 150)

# Chat Input (Stored in session state)
input_query = st.chat_input("Describe the wildfire situation")

if input_query:
    st.session_state.query = input_query

# Run Pipeline If Query Exists
if st.session_state.query:
    with st.chat_message("user"):
        st.markdown(st.session_state.query)

    # Combine query + weather info
    weather_info = f"""
Date of Fire Fighting Approach: {reqdate}
Time of Fire Fighting Approach: {reqtime}
Weather: {sky}
Temperature: {temp} °C
Humidity: {humid} %
Wind Direction: {winddir}
Wind Speed: {windspeed} m/s
"""
    full_query = f"{st.session_state.query}\n\n--- WEATHER CONDITIONS ---\n{weather_info}"

    # Run FireGPT pipeline:
    json_handler = JsonHandler("./data/processed/chunking_metadata.json")
    retriever = Retrieve(
        query=full_query,
        chunks=json_handler.dataset_str,
        json_chunks=json_handler.dataset_json,
        top_k=3,
        model_name="models/all-MiniLM-L6-v2",
        index_path="models/index.faiss",
    )
    rag = RAGPipeline(model_path=MODEL_PATHS[selected_model], n_gpu_layers=n_gpu_layers, context_length=context_length)
    response = rag.run(full_query, retriever.results)

    with st.chat_message("assistant"):
        st.markdown(response)
