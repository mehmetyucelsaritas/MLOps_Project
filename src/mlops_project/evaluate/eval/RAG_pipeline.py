import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops_project.pdf_converter import ConvertPdf2Text
from mlops_project.chunking import Chunking
from mlops_project.embedding import Embedding
from mlops_project.retrieve import Retrieve
from RAG_LLM.rag_runner import RAGPipeline


def setup_pipeline(pdf_path, txt_path, model, max_words, min_words):
    ConvertPdf2Text(pdf_path, txt_path)
    chunker = Chunking(txt_path, max_words, min_words)
    embedder = Embedding(chunker.chunks, model)
    return chunker.chunks, embedder


def run_rag_pipeline(query, chunks, embedder, top_k=3):
    retriever = Retrieve(query, chunks, top_k, embedder.model, embedder.index)
    rag = RAGPipeline(query, retriever.results)
    response = rag.run()
    return {
        "user_input": query,
        "response": response,
        "retrieved_contexts": retriever.results,
    }
