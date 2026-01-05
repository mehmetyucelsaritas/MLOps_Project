import os
import faiss
from mlops_project.utility import Parser, JsonHandler
from mlops_project.model.embedding import Embedding
from mlops_project.model.retrieve import Retrieve
# from RAG_LLM.rag_runner import RAGPipeline


def main():
    # Flag to decide whether to update (rebuild) the FAISS index database
    update_index = False 
    # Initialize argument parser and parse command line arguments
    arg_parser = Parser()
    args = arg_parser.args

    # -------------------- PIPELINE EXECUTION -------------------------

    # Load metadata from JSON file using JsonHandler
    json_handler = JsonHandler(args.metadata_path)

    # If update_index is True, generate new embeddings and rebuild the index database
    if update_index:
        embedder = Embedding(json_handler.dataset_str, args.model)
        faiss.write_index(embedder.index, f"{args.index_path}")
        print(f"index database saved to {args.index_path}")

    # Create a retriever object which handles querying with given parameters,    
    retriever = Retrieve(
        args.query, 
        json_handler.dataset_str, 
        json_handler.dataset_json, 
        args.top_k, 
        args.model, 
        args.index_path
    )
    
    # Outputs
    print(f"[QUERY]: {args.query}\n")
    for i, chunk in enumerate(retriever.results):
        print(f"[RESULT {i+1}]:\n{chunk}\n")

    # rag = RAGPipeline(
    #     model_path=args.model_path,
    #     n_gpu_layers=args.n_gpu_layers,
    #     context_length=args.context_length
    # )
    # response = rag.run(args.query, retriever.results)
    # print("\n[LLM RESPONSE]:\n", response)

if __name__ == "__main__":
    main()
