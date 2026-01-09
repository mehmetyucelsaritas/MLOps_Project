import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any  # Added Dict and Any types

from fastapi import FastAPI, HTTPException
from hydra import compose, initialize
from loguru import logger
from pydantic import BaseModel

# Import your existing project modules
from mlops_project.model.retrieve import Retrieve
from mlops_project.rag_llm.rag_runner import RAGPipeline
from mlops_project.utility import JsonHandler, Parser

# --- Global State Dictionary ---
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle startup and shutdown events.
    """
    logger.info("Loading RAG Pipeline resources...")

    # 1. Initialize Hydra Configuration
    try:
        with initialize(version_base="1.3", config_path="../../configs"):
            # Provide job_name="api" to avoid Hydra mode issues if necessary
            config = compose(config_name="default_config")
    except Exception as e:
        logger.error(f"Failed to load Hydra configuration: {e}")
        raise e

    # 2. Parse Arguments
    original_argv = sys.argv
    sys.argv = ["api.py"]

    try:
        # Now Parser will only use defaults + config, ignoring uvicorn flags
        arg_parser = Parser(config)
        args = arg_parser.args
    except Exception as e:
        logger.error(f"Failed to parse arguments: {e}")
        raise e
    finally:
        # Restore original args
        sys.argv = original_argv

    # 3. Load Metadata
    try:
        json_handler = JsonHandler(args.metadata_path)
    except Exception as e:
        logger.error(f"Failed to load metadata at {args.metadata_path}: {e}")
        raise e

    # 4. Initialize LLM (RAGPipeline)
    rag_pipeline = RAGPipeline(model_path=args.model_path, n_gpu_layers=args.n_gpu_layers, context_length=args.context_length)

    # Store everything in resources
    resources["config"] = config
    resources["args"] = args
    resources["json_handler"] = json_handler
    resources["rag_pipeline"] = rag_pipeline

    logger.info("RAG Pipeline resources loaded successfully.")

    yield

    resources.clear()
    logger.info("Resources cleared.")


# --- FastAPI App ---
app = FastAPI(title="RAG Inference API", lifespan=lifespan)


# --- Pydantic Models ---
class InferenceRequest(BaseModel):
    query: str


class InferenceResponse(BaseModel):
    response: str
    retrieved_chunks: List[Dict[str, Any]]


# --- Endpoints ---
@app.get("/health")
def health_check():
    status = "ok" if "rag_pipeline" in resources else "loading"
    return {"status": status}


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    try:
        # Retrieve loaded resources
        args = resources.get("args")
        json_handler = resources.get("json_handler")
        rag_pipeline = resources.get("rag_pipeline")

        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        query = request.query
        logger.info(f"[API QUERY]: {query}")

        # 1. Retrieval Step
        retriever = Retrieve(query, json_handler.dataset_str, json_handler.dataset_json, args.top_k, args.model, args.index_path)

        # 2. Generation Step
        llm_response = rag_pipeline.run(query, retriever.results)

        logger.info("[LLM RESPONSE GENERATED]")

        return InferenceResponse(response=llm_response, retrieved_chunks=retriever.results)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
