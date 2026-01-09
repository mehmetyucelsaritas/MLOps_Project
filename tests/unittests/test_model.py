import os
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from mlops_project.model.chunking import Chunking
from mlops_project.model.embedding import Embedding
from mlops_project.model.retrieve import Retrieve

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def sample_text(tmp_path):
    """
    Creates a temporary text file with invisible characters.
    """
    text = "This is the first sentence.\nThis\u200b is the second sentence.\tThis is the third sentence."

    file_path = tmp_path / "doc1.txt"
    file_path.write_text(text, encoding="utf-8")
    return file_path


@pytest.fixture
def fake_sent_tokenize(monkeypatch):
    """
    Mock sent_tokenize to avoid NLTK dependency.
    """

    def mock_tokenize(text):
        return ["This is the first sentence.", "This is the second sentence.", "This is the third sentence."]

    monkeypatch.setattr("mlops_project.model.chunking.sent_tokenize", mock_tokenize)


@pytest.fixture
def task1_metadata():
    return {
        "doc1.pdf": {
            "topic": "AI",
            "region": "EU",
            "source_name": "Test Source",
            "language": "en",
            "summary": "Test summary",
        }
    }


# -----------------------------
# Tests: read_output_file
# -----------------------------


def test_read_output_file_removes_invisible_chars(sample_text, fake_sent_tokenize):
    chunker = Chunking(
        text_path=str(sample_text),
        max_words=50,
        min_words=5,
    )

    assert "\n" not in chunker.content
    assert "\u200b" not in chunker.content
    assert "\t" not in chunker.content
    assert "This is the second sentence." in chunker.content


# -----------------------------
# Tests: chunk_text
# -----------------------------


def test_chunk_text_basic(fake_sent_tokenize, sample_text):
    chunker = Chunking(
        text_path=str(sample_text),
        max_words=10,
        min_words=3,
    )

    assert isinstance(chunker.chunks, list)
    assert len(chunker.chunks) >= 1

    for chunk in chunker.chunks:
        assert isinstance(chunk, str)
        assert len(chunk.split()) <= 10


def test_chunk_text_long_sentence(monkeypatch, sample_text):
    def mock_long_sentence(text):
        return ["word " * 50]  # 50-word sentence

    monkeypatch.setattr("mlops_project.model.chunking.sent_tokenize", mock_long_sentence)

    chunker = Chunking(
        text_path=str(sample_text),
        max_words=10,
        min_words=3,
    )

    assert len(chunker.chunks) == 1
    assert len(chunker.chunks[0].split()) > 10


# -----------------------------
# Tests: generate_metadata
# -----------------------------


def test_generate_metadata_without_task1(sample_text, fake_sent_tokenize):
    chunker = Chunking(
        text_path=str(sample_text),
        max_words=10,
        min_words=3,
        task1_metadata=None,
    )

    assert isinstance(chunker.chunk_items, list)
    assert "text" in chunker.chunk_items[0]
    assert len(chunker.chunk_items[0]) == 1  # only "text"


def test_generate_metadata_with_task1(sample_text, fake_sent_tokenize, task1_metadata):
    chunker = Chunking(
        text_path=str(sample_text),
        max_words=10,
        min_words=3,
        task1_metadata=task1_metadata,
    )

    item = chunker.chunk_items[0]

    assert item["source"] == "doc1.pdf"
    assert item["topic"] == "AI"
    assert item["region"] == "EU"
    assert item["language"] == "en"
    assert item["summary"] == "Test summary"


# -----------------------------
# Integration test
# -----------------------------


def test_full_chunking_pipeline(sample_text, fake_sent_tokenize, task1_metadata):
    chunker = Chunking(
        text_path=str(sample_text),
        max_words=10,
        min_words=3,
        task1_metadata=task1_metadata,
    )

    assert chunker.content
    assert chunker.chunks
    assert chunker.chunk_items
    assert len(chunker.chunks) == len(chunker.chunk_items)


@pytest.fixture
def dummy_chunks():
    return ["This is the first chunk.", "This is the second chunk.", "Another text chunk."]


@pytest.fixture
def dummy_embeddings():
    # 3 chunks, 5-dim embeddings
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.9, 0.8, 0.7, 0.6, 0.5],
        ],
        dtype=np.float32,
    )


@patch("mlops_project.model.embedding.SentenceTransformer")
def test_embedding_initialization(
    mock_sentence_transformer,
    dummy_chunks,
    dummy_embeddings,
):
    """
    Test that Embedding initializes model, embeddings, and FAISS index correctly.
    """

    # Mock model instance
    mock_model = MagicMock()
    mock_model.encode.return_value = dummy_embeddings
    mock_sentence_transformer.return_value = mock_model

    model_name = "fake-model"

    embedding = Embedding(chunks=dummy_chunks, model_name=model_name)

    # SentenceTransformer initialized correctly
    mock_sentence_transformer.assert_called_once_with(model_name)

    # encode called with correct args
    mock_model.encode.assert_called_once_with(dummy_chunks, show_progress_bar=True)

    # embeddings correctness
    assert isinstance(embedding.embeddings, np.ndarray)
    assert embedding.embeddings.shape == (3, 5)

    # FAISS index correctness
    assert isinstance(embedding.index, faiss.IndexFlatL2)
    assert embedding.index.d == 5  # embedding dimension
    assert embedding.index.ntotal == 3  # number of vectors


@patch("mlops_project.model.embedding.SentenceTransformer")
def test_faiss_search_works(
    mock_sentence_transformer,
    dummy_chunks,
    dummy_embeddings,
):
    """
    Ensure FAISS index can perform a similarity search.
    """

    mock_model = MagicMock()
    mock_model.encode.return_value = dummy_embeddings
    mock_sentence_transformer.return_value = mock_model

    embedding = Embedding(chunks=dummy_chunks, model_name="fake-model")

    query = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)

    distances, indices = embedding.index.search(query, k=1)

    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)
    assert indices[0][0] in [0, 1, 2]


def test_retrieve_search_index_simple(monkeypatch):
    # -----------------------
    # Arrange
    # -----------------------
    query = "test query"
    chunks = ["chunk1", "chunk2", "chunk3"]
    json_chunks = ["json1", "json2", "json3"]
    top_k = 2
    model_name = "fake-model"
    index_path = "fake.index"

    # ---- Mock FAISS index ----
    mock_index = MagicMock()

    # FAISS search returns (distances, indices)
    mock_index.search.return_value = (
        np.array([[0.1, 0.2]]),
        np.array([[2, 0]]),
    )

    # Patch faiss.read_index
    monkeypatch.setattr(
        "mlops_project.model.retrieve.faiss.read_index",
        lambda _: mock_index,
    )

    # ---- Mock SentenceTransformer ----
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.5, 0.5]])

    monkeypatch.setattr(
        "mlops_project.model.retrieve.SentenceTransformer",
        lambda _: mock_model,
    )

    # -----------------------
    # Act
    # -----------------------
    retriever = Retrieve(
        query=query,
        chunks=chunks,
        json_chunks=json_chunks,
        top_k=top_k,
        model_name=model_name,
        index_path=index_path,
    )

    # -----------------------
    # Assert
    # -----------------------
    # Correct calls
    mock_model.encode.assert_called_once_with([query])
    mock_index.search.assert_called_once()

    # Correct results (index order: [2, 0])
    assert retriever.results == ["json3", "json1"]
    assert len(retriever.results) == top_k
