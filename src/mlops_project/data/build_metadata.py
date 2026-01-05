import json
import os

from mlops_project.model.chunking import Chunking


def main():
    txt_folder = "./data/processed"
    task1_metadata_path = "./data/raw/all_metadata.json"
    output_metadata_path = "./data/processed/chunking_metadata.json"

    with open(task1_metadata_path, "r", encoding="utf-8") as f:
        task1_metadata = json.load(f)

    task1_table = {entry["filename"]: entry for entry in task1_metadata}

    task2_metadata = []

    for fname in os.listdir(txt_folder):
        if fname.endswith(".txt"):
            txt_path = os.path.join(txt_folder, fname)
            chunks = Chunking(txt_path, max_words=300, min_words=100, task1_metadata=task1_table)

            task2_metadata.extend(chunks.chunk_items)

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(task2_metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
