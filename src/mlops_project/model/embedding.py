import faiss  # Store embedded ckunks as database
import numpy as np
from sentence_transformers import SentenceTransformer  # Embed chunks to vector


class Embedding:
    """
    A class for embedding text chunks using a SentenceTransformer model
    and storing them in a FAISS index for efficient similarity search.

    Attributes:
        chunks (List[str]): List of text chunks to be embedded.
        model (SentenceTransformer): Pretrained sentence embedding model.
        embeddings (np.ndarray): Numpy array of vectorized chunks.
        index (faiss.Index): FAISS index for similarity search.
    """

    def __init__(self, chunks: list[str], model_name: str):
        self.model = SentenceTransformer(model_name)  # Or any model you choose
        self.chunks = chunks
        self.embeddings = self.embed_chunks()
        self.index = self.build_faiss_index()

    def embed_chunks(self):
        """
        Converts the list of text chunks into vector embeddings.

        Returns:
            np.ndarray: Embeddings of the chunks as a numpy array.
        """

        return self.model.encode(self.chunks, show_progress_bar=True)

    def build_faiss_index(self):
        """
        Builds a FAISS index from the embedded chunks.

        Returns:
            faiss.IndexFlatL2: A flat L2 FAISS index with the embedded vectors.
        """

        dim = len(self.embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(self.embeddings))
        return index
