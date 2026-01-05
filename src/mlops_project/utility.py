import argparse
import json
from typing import Any, Dict, List


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Semantic PDF search using embedding and retrieval.")

        # File paths
        # parser.add_argument("--txt_path", default="../out/output.txt", help="Path to save extracted text.")
        self.parser.add_argument("--index_path", default="models/index.faiss", help="Path to save index database.")
        self.parser.add_argument(
            "--tokenizer_path", default="models/nltk_data/tokenizers", help="Path to save tokenizer models."
        )
        self.parser.add_argument(
            "--metadata_path", default="data/processed/chunking_metadata.json", help="Path to save chunking database"
        )
        # Query
        self.parser.add_argument(
            "--query",
            default="There is a wildfire approaching the south side of the city,"
            " threatening homes and a nearby gas station. Winds are strong, and visibility is limited.",
            help="The query string to search for.",
        )

        # Parameters
        self.parser.add_argument("--max_words", type=int, default=250, help="Maximum number of words per chunk.")
        self.parser.add_argument("--min_words", type=int, default=100, help="Minimum number of words per chunk.")
        self.parser.add_argument("--top_k", type=int, default=3, help="Number of top similar chunks to return.")
        self.parser.add_argument(
            "--model", type=str, default="models/all-MiniLM-L6-v2", help="Name of the sentence transformer model."
        )

        self.parser.add_argument(
            "--model_path", type=str, default="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", help="Path to GGUF model"
        )
        self.parser.add_argument("--n_gpu_layers", type=int, default=-1, help="Number of GPU layers to offload")
        self.parser.add_argument("--context_length", type=int, default=2048, help="Max context length for LLM")

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
