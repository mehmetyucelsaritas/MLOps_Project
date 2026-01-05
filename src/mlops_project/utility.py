import argparse
import json
from typing import Any, Dict, List


class Parser:
    def __init__(self, config):
        self.parser = argparse.ArgumentParser(description="Semantic PDF search using embedding and retrieval.")
        
        parameters = config.experiment
        # File paths
        self.parser.add_argument("--index_path", default=parameters["index_path"], help="Path to save index database.")
        self.parser.add_argument("--tokenizer_path", default=parameters["tokenizer_path"], help="Path to save tokenizer models.")
        self.parser.add_argument("--metadata_path", default=parameters["metadata_path"], help="Path to save chunking database")
        # Query
        self.parser.add_argument("--query", default=parameters["query"], help="The query string to search for.")

        # Parameters
        self.parser.add_argument("--max_words", type=int, default=parameters["max_words"], help="Maximum number of words per chunk.")
        self.parser.add_argument("--min_words", type=int, default=parameters["min_words"], help="Minimum number of words per chunk.")
        self.parser.add_argument("--top_k", type=int, default=parameters["top_k"], help="Number of top similar chunks to return.")
        self.parser.add_argument("--model", type=str, default=parameters["model"], help="Name of the sentence transformer model.")

        self.parser.add_argument("--model_path", type=str, default=parameters["model_path"], help="Path to GGUF model")
        self.parser.add_argument("--n_gpu_layers", type=int, default=parameters["n_gpu_layers"], help="Number of GPU layers to offload")
        self.parser.add_argument("--context_length", type=int, default=parameters["context_length"], help="Max context length for LLM")

        self.args = self.parser.parse_args()


class JsonHandler:
    """
    A utility class for loading a JSON file consisting of a list of dictionaries,
    and extracting the values associated with the "text" key.

    Attributes:
        metadata_path (str): Path to the JSON metadata file.
        dataset_json (List[Dict[str, Any]]): Parsed content of the JSON file.
        dataset_str (List[str]): List of text strings extracted from the JSON file.
    """

    def __init__(self, metadata_path: str):
        """
        Initializes the JsonHandler with a path to the JSON file.

        Args:
            metadata_path (str): Path to the JSON file containing metadata.
        """
        self.metadata_path = metadata_path
        self.dataset_json = []
        self.dataset_str = []

        self._load_json_file()
        self._extract_text_entries()

    def _load_json_file(self) -> None:
        """
        Opens and loads the JSON file into memory.
        """
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                self.dataset_json = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at: {self.metadata_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from file: {self.metadata_path}")

    def _extract_text_entries(self) -> None:
        """
        Extracts the 'text' field from each dictionary in the dataset
        and stores them in the dataset_str list.
        """
        for entry in self.dataset_json:
            if "text" in entry:
                self.dataset_str.append(entry["text"])
            else:
                raise KeyError(f"Missing 'text' key in entry: {entry}")
