import faiss  # Store embedded ckunks as database
import numpy as np
from sentence_transformers import SentenceTransformer  # Embed chunks to vector


class Retrieve:
    """
    A class to perform similarity-based retrieval on embedded text chunks using FAISS.

    Attributes:
        query (str): The input query string.
        chunks (List[str]): The original text chunks used to build the index.
        top_k (int): Number of top similar results to return.
        model (Embedding): Instance of the Embedding class for query encoding.
        index (faiss.Index): FAISS index containing chunk embeddings.
        results (List[str]): Top-k most relevant chunks for the query.
    """

    def __init__(self, query: str, chunks: list, json_chunks: list, top_k: int, model_name: str, index_path: str) -> list:
        self.top_k = top_k
        self.index = faiss.read_index(index_path)
        self.query = query
        self.model = SentenceTransformer(model_name)  # Or any model you choose
        self.chunks = chunks
        self.json_chunks = json_chunks
        self.results = self.search_index_simple()

    def search_index_simple(self):
        """
        Encodes the query and retrieves the top-k most similar chunks from the FAISS index.

        Returns:
            List[str]: The list of top-k relevant text chunks.
        """

        query_vec = self.model.encode([self.query])
        d, i = self.index.search(np.array(query_vec), self.top_k)

        results = []
        for idx in i[0]:
            results.append(self.json_chunks[idx])  # Directly use chunk text

        return results
