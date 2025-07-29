"""
Data Processing Module
----------------------
1. Cleans and deduplicates raw text data
2. Normalizes text
3. Tokenizes text (basic)
4. Saves processed output

Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""
import re
import os
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_files(file_paths: List[str]) -> List[str]:
    """Load raw text files into memory."""
    texts = []
    for file_path in tqdm(file_paths, desc="Loading raw data"):
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def clean_text(text: str) -> str:
    """Clean text: remove unwanted chars, multiple spaces."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?-]", "", text)
    return text.strip()


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove duplicate entries."""
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def tokenize_text(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def process_and_save(raw_files: List[str], output_file: str = "processed.csv") -> str:
    """Load, clean, deduplicate, tokenize, and save data."""
    raw_texts = load_raw_files(raw_files)
    cleaned = [clean_text(t) for t in raw_texts]
    unique = deduplicate_texts(cleaned)

    df = pd.DataFrame(unique, columns=["text"])
    df["tokens"] = df["text"].apply(tokenize_text)

    output_path = PROCESSED_DIR / output_file
    df.to_csv(output_path, index=False)

    print(f"[INFO] Processed data saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    raw_files = [
        "data/raw/web_data.txt",
        "data/raw/pdf_data.txt"
    ]
    raw_files = [f for f in raw_files if os.path.exists(f)]

    if not raw_files:
        print("[WARN] No raw files found. Run data_collection first.")
    else:
        process_and_save(raw_files)
