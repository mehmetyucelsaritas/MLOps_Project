import os

import nltk  # Chunk the text
from nltk.tokenize import sent_tokenize

# sent_tokenize()

# Characters to be removed.
invisible_chars = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte order mark
    "\x0c",  # non-breaking space
    "\n",  # new line
    "\xad",  # do not know
    "\t",  # tab
    "•",  # bullet
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
NLTK_DATA_PATH = os.path.join(REPO_ROOT, "..", "models", "nltk_data")


class Chunking:
    """
    A class for processing and chunking text into manageable sections based on word count.

    Attributes:
        text_path (str): The file path to the text file to be chunked.
        max_words (int): Maximum number of words allowed per chunk.
        min_words (int): Minimum number of words per chunk (if possible).
        content (str): The cleaned content of the text file.
        chunks (List[str]): A list of text chunks produced from the content.
    """

    def __init__(self, text_path: str, max_words: int, min_words: int, task1_metadata=None) -> None:
        nltk.data.path.insert(0, NLTK_DATA_PATH)
        # nltk.download('punkt_tab')
        self.text_path = text_path
        self.max_words = max_words
        self.min_words = min_words
        # self.file_name = "output.txt"
        self.content = self.read_output_file()
        self.chunks = self.chunk_text(self.content, self.max_words, self.min_words)
        self.chunk_items = self.generate_metadata(task1_metadata)

    def read_output_file(self):
        """
        Reads the text file from the given path and removes unwanted invisible characters.

        Returns:
            str: Cleaned content of the text file.
        """

        with open(self.text_path, "r", encoding="utf-8") as file:
            content = file.read()

        for char in invisible_chars:
            content = content.replace(char, "")

        return content

    def chunk_text(self, text: str, max_words: int = 300, min_words: int = 100):
        """
        Splits the text into chunks based on word count, attempting to keep chunks
        within the [min_words, max_words] range when possible.

        Args:
            text (str): The full text to be chunked.
            max_words (int): Maximum words per chunk.
            min_words (int): Minimum words per chunk.

        Returns:
            List[str]: List of text chunks.
        """

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If sentence is too long, add it as its own chunk
            if sentence_words > max_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(sentence)  # Long sentence as standalone
                continue

            if current_len + sentence_words <= max_words:
                current_chunk.append(sentence)
                current_len += sentence_words
            else:
                if current_len >= min_words:  #
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_len = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_len += sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_metadata(self, task1_metadata):
        if task1_metadata is None:
            return [{"text": chunk} for chunk in self.chunks]

        pdf_filename = os.path.basename(self.text_path).replace(".txt", ".pdf")
        task1_entry = task1_metadata.get(pdf_filename, {})

        metadata_chunks = []
        for i, chunk in enumerate(self.chunks):
            metadata_chunks.append(
                {
                    "text": chunk,
                    "source": pdf_filename,
                    "topic": task1_entry.get("topic", ""),
                    "region": task1_entry.get("region", ""),
                    "source_name": task1_entry.get("source_name", ""),
                    "language": task1_entry.get("language", ""),
                    "summary": task1_entry.get("summary", ""),
                }
            )
        return metadata_chunks
