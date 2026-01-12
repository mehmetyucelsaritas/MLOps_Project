import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings as LC_HuggingFaceEmbeddings

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness, ResponseRelevancy

from dotenv import load_dotenv
import json
import datetime
import time

from mlops_project.utility import JsonHandler
from mlops_project.model.retrieve import Retrieve
from mlops_project.rag_llm.rag_runner import RAGPipeline


# Load environment variables from .env file
load_dotenv()

# Now the API key will be available in os.environ
api_key = os.getenv("GOOGLE_API_KEY")

config = {
    "model": "gemini-2.0-flash-lite",  # or other model IDs
    "temperature": 0.4,
    "max_tokens": None,
    "top_p": 0.8,
}

# Initialize with Google AI Studio
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
    )
)

# Model mapping
MODEL_PATHS = {"Mistral-7B": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", "Phi-3": "models/Phi-3-mini-4k-instruct-q4.gguf", "TinyLlama": "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"}

# Load testset
testset_path = os.path.join(os.getcwd(), "testset", "test_query.json")

with open(testset_path, "r", encoding="utf-8") as f:
    testset = json.load(f)

# Comment from here:
# Set up RAG pipeline once
dataset = []
timing_records = []
for item in testset:
    query = item["user_input"]
    json_handler = JsonHandler("out/chunking_metadata.json")
    retriever = Retrieve(query=query, chunks=json_handler.dataset_str, json_chunks=json_handler.dataset_json, top_k=3, model_name="models/all-MiniLM-L6-v2", index_path="out/index.faiss")
    rag = RAGPipeline(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_gpu_layers=0, context_length=4096)
    start_time = time.time()
    response = rag.run(query, retriever.results)
    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    dataset.append({"user_input": query, "retrieved_contexts": [ctx["text"] for ctx in retriever.results if "text" in ctx], "response": response})

    timing_records.append({"user_input": query, "response_time_sec": response_time})

evaluation_dataset = EvaluationDataset.from_list(dataset)
lc_embed = LC_HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = LangchainEmbeddingsWrapper(embeddings=lc_embed)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextPrecisionWithoutReference(), Faithfulness(), ResponseRelevancy()],
    llm=evaluator_llm,
    embeddings=embedding_model,
)
print(result)

# create folder to store results
results_dir = os.path.join(os.getcwd(), "eval", "eval_results")
os.makedirs(results_dir, exist_ok=True)

df = result.to_pandas()
today = datetime.datetime.now().strftime("%Y-%m-%d")
df.to_json(
    os.path.join(results_dir, f"evaluation_results_{today}_M7.json"),
    indent=2,
    force_ascii=False,
)

# Save timing records
timing_path = os.path.join(results_dir, f"response_times_{today}_M7.json")
with open(timing_path, "w", encoding="utf-8") as tf:
    json.dump(timing_records, tf, indent=2, ensure_ascii=False)
# To here.

# Uncomment to iterate MODEL_PATHS
# for model_name, model_path in MODEL_PATHS.items():
#     print(f"\n🧪 Evaluating with model: {model_name}\n")

#     dataset = []
#     for item in testset:
#         query = item["user_input"]
#         json_handler = JsonHandler("out/chunking_metadata.json")
#         retriever = Retrieve(
#             query=query,
#             chunks=json_handler.dataset_str,
#             json_chunks=json_handler.dataset_json,
#             top_k=3,
#             model_name="models/all-MiniLM-L6-v2",
#             index_path="out/index.faiss"
#         )

#         rag = RAGPipeline(
#             model_path=model_path,
#             n_gpu_layers=0,
#             context_length=4096
#         )
#         response = rag.run(query, retriever.results)
#         dataset.append({
#             "question": query,
#             "contexts": [ctx["text"] for ctx in retriever.results if "text" in ctx],
#             "answer": response
#         })

#     # Evaluate this model's results
#     evaluation_dataset = EvaluationDataset.from_list(dataset)
#     lc_embed = LC_HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     embedding_model = LangchainEmbeddingsWrapper(embeddings=lc_embed)

#     result = evaluate(
#         dataset=evaluation_dataset,
#         metrics=[
#             LLMContextPrecisionWithoutReference(),
#             Faithfulness(),
#             ResponseRelevancy()
#         ],
#         llm=evaluator_llm,
#         embeddings=embedding_model,
#     )

#     print(f"📊 Results for {model_name}:", result)

#     # Save result
#     df = result.to_pandas()
#     today = datetime.datetime.now().strftime("%Y-%m-%d")
#     results_dir = os.path.join(os.getcwd(), "eval", "eval_results")
#     os.makedirs(results_dir, exist_ok=True)

#     df.to_json(
#         os.path.join(results_dir, f"evaluation_results_{model_name}_{today}.json"),
#         indent=2,
#         force_ascii=False,
#     )
