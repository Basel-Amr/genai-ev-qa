"""
Test Data Processing Module
---------------------------
Ensures cleaning, deduplication, and tokenization works.

Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""
import os
import pandas as pd
import pytest
from pathlib import Path
from src.data_processing import clean_text, deduplicate_texts, tokenize_text, process_and_save

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_FILE = "data/processed/test_processed.csv"

@pytest.fixture
def sample_raw_files(tmp_path):
    # Create sample raw text files
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("Hello Electric Vehicle!\nCharging stations are cool.")
    file2.write_text("Hello Electric Vehicle!\nCharging stations are cool.")
    return [str(file1), str(file2)]


def test_clean_text():
    text = "Hello   World!!!"
    cleaned = clean_text(text)
    assert cleaned == "hello world"


def test_deduplicate_texts():
    texts = ["hello world", "hello world", "new data"]
    result = deduplicate_texts(texts)
    assert len(result) == 2


def test_tokenize_text():
    tokens = tokenize_text("electric vehicle charging")
    assert tokens == ["electric", "vehicle", "charging"]


def test_process_and_save(sample_raw_files):
    output = process_and_save(sample_raw_files, output_file="test_processed.csv")
    assert os.path.exists(output)
    df = pd.read_csv(output)
    assert "text" in df.columns and "tokens" in df.columns
    assert len(df) > 0
