import os
import re

import pdfplumber


def fix_broken_words(text):
    """
    Merge incorrectly split words, e.g., 'f ires' -> 'fires', while preserving real spacing.
    """
    # Fix words with single spaces inside them (e.g., "f ires")
    text = re.sub(r"\b(?:[a-zA-Z]\s){1,4}[a-zA-Z]\b", lambda m: m.group(0).replace(" ", ""), text)

    # fix: 'ont he' → 'on the'
    text = re.sub(
        r"\b(\w{1,3})\s+(\w{1,3})\b",
        lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) <= 5 else m.group(0),
        text,
    )

    return text


def extract_clean_text_from_pdf(pdf_path, margin=50):
    """
    Extracts cleaned text from a PDF, removing headers, footers, and page numbers.
    margin: the top/bottom height (in pts) to skip likely header/footer areas.
    """
    cleaned_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_height = page.height
            words = page.extract_words(use_text_flow=True)

            # Filter out header/footer words by top position
            body_words = [w["text"] for w in words if margin < w["top"] < (page_height - margin)]
            page_text = " ".join(body_words).strip()

            # Regex cleanup for things like "Page X" or numbers
            import re

            page_text = re.sub(r"\bPage\s*\d+\b", "", page_text)
            page_text = re.sub(r"^\s*\d+\s*$", "", page_text, flags=re.MULTILINE)
            cleaned_text.append(page_text)

    full_text = "\n".join(cleaned_text)
    final_text = fix_broken_words(full_text)
    return final_text


def save_text_to_file(text, output_path):
    """Save extracted text to a text file."""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


def process_pdf_files(dataset_folder, processed_folder):
    """Process all PDF files in the dataset folder."""
    metadata = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(dataset_folder, filename)
            text = extract_clean_text_from_pdf(pdf_path)
            text_filename = filename.replace(".pdf", ".txt")
            text_path = os.path.join(processed_folder, text_filename)
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            save_text_to_file(text, text_path)


if __name__ == "__main__":
    dataset_folder = "data/raw"
    processed_folder = "data/processed"
    if os.path.exists(dataset_folder):
        process_pdf_files(dataset_folder, processed_folder)
    else:
        print(f"Dataset folder '{dataset_folder}' does not exist.")
