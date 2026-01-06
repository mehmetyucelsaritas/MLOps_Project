import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

import mlops_project.data.build_metadata as build_metadata
from mlops_project.data.data import extract_clean_text_from_pdf, fix_broken_words, process_pdf_files, save_text_to_file


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("ont he table", "onthe table"),  # <= 5 chars → merged
        ("this is fine", "this is fine"),
    ],
)
def test_fix_broken_words(input_text, expected):
    assert fix_broken_words(input_text) == expected


@patch("mlops_project.data.data.pdfplumber.open")
def test_extract_clean_text_from_pdf(mock_open):
    mock_page = mock_pdf_page(
        words=[
            {"text": "Header", "top": 10},  # header (ignored)
            {"text": "This", "top": 200},
            {"text": "fires", "top": 200},
            {"text": "Page", "top": 790},  # footer
            {"text": "1", "top": 790},
        ]
    )

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_open.return_value.__enter__.return_value = mock_pdf

    result = extract_clean_text_from_pdf("dummy.pdf", margin=50)

    assert "fires" in result
    assert "Header" not in result
    assert "Page" not in result


def test_save_text_to_file(tmp_path: Path):
    output_file = tmp_path / "output.txt"
    content = "Hello world"

    save_text_to_file(content, output_file)

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == content


@patch("mlops_project.data.data.extract_clean_text_from_pdf")
@patch("mlops_project.data.data.save_text_to_file")
@patch("os.listdir")
def test_process_pdf_files(mock_listdir, mock_save, mock_extract, tmp_path):
    mock_listdir.return_value = ["file1.pdf", "file2.txt"]
    mock_extract.return_value = "clean text"

    dataset = tmp_path / "raw"
    processed = tmp_path / "processed"
    dataset.mkdir()

    process_pdf_files(dataset, processed)

    mock_extract.assert_called_once()
    mock_save.assert_called_once()


def test_build_metadata_creates_chunking_metadata(tmp_path, monkeypatch, fake_metadata, mock_chunking):
    # --- Arrange ---
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"

    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    # fake txt file
    (processed_dir / "doc1.txt").write_text("Some text", encoding="utf-8")
    (processed_dir / "ignore.pdf").write_text("Should be ignored", encoding="utf-8")

    # fake task1 metadata
    task1_metadata_path = raw_dir / "all_metadata.json"
    task1_metadata_path.write_text(json.dumps(fake_metadata), encoding="utf-8")

    output_path = processed_dir / "chunking_metadata.json"

    # patch paths inside build_metadata
    monkeypatch.setattr(build_metadata, "os", build_metadata.os)
    monkeypatch.chdir(tmp_path)

    # --- Act ---
    build_metadata.main()

    # --- Assert ---
    assert output_path.exists()

    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(output) == 2
    assert output[0]["text"] == "chunk 1"

    # Chunking called once for txt file
    mock_chunking.assert_called_once()
    args, kwargs = mock_chunking.call_args

    assert args[0].endswith("doc1.txt")

    assert kwargs["max_words"] == 300
    assert kwargs["min_words"] == 100
    assert isinstance(kwargs["task1_metadata"], dict)


@pytest.fixture
def fake_metadata():
    return [
        {
            "filename": "doc1.pdf",
            "topic": "AI",
            "region": "EU",
            "source_name": "TestSource",
            "language": "en",
            "summary": "summary1",
        }
    ]


@pytest.fixture
def mock_chunking(monkeypatch):
    """
    Mock Chunking so we don't test chunking logic here.
    """
    mock_instance = MagicMock()
    mock_instance.chunk_items = [
        {"text": "chunk 1", "source": "doc1.pdf"},
        {"text": "chunk 2", "source": "doc1.pdf"},
    ]

    mock_class = MagicMock(return_value=mock_instance)

    monkeypatch.setattr(build_metadata, "Chunking", mock_class)
    return mock_class


@staticmethod
def mock_pdf_page(words, height=800):
    page = MagicMock()
    page.height = height
    page.extract_words.return_value = words
    return page
