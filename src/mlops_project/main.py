import os

import faiss
import hydra
from loguru import logger

from mlops_project.model.embedding import Embedding
from mlops_project.model.retrieve import Retrieve
from mlops_project.rag_llm.rag_runner import RAGPipeline
from mlops_project.utility import JsonHandler, Parser


@hydra.main(version_base="1.3", config_path="../../configs", config_name="default_config.yaml")
def main(config):
    # Flag to decide whether to update (rebuild) the FAISS index database
    update_index = False
    # Initialize argument parser and parse command line arguments
    arg_parser = Parser(config)
    args = arg_parser.args

    # -------------------- PIPELINE EXECUTION -------------------------
    # log information to hydra experiments
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "my_logger_hydra.log"))
    logger.info(f"Configuration details: {config}")

    # Load metadata from JSON file using JsonHandler
    json_handler = JsonHandler(args.metadata_path)

    # If update_index is True, generate new embeddings and rebuild the index database
    if update_index:
        embedder = Embedding(json_handler.dataset_str, args.model)
        faiss.write_index(embedder.index, f"{args.index_path}")
        logger.info(f"index database saved to path {args.index_path}\n")

    # Create a retriever object which handles querying with given parameters,
    retriever = Retrieve(
        args.query, json_handler.dataset_str, json_handler.dataset_json, args.top_k, args.model, args.index_path
    )

    # Outputs
    logger.info(f"[QUERY]: {args.query}\n")
    for i, chunk in enumerate(retriever.results):
        logger.info(f"[RESULT {i + 1}]:\n{chunk}\n")

    rag = RAGPipeline(model_path=args.model_path, n_gpu_layers=args.n_gpu_layers, context_length=args.context_length)
    response = rag.run(args.query, retriever.results)
    logger.info(f"\n[LLM RESPONSE]:\n {response}")


if __name__ == "__main__":
    main()
