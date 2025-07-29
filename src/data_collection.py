"""
Data Collection Module
----------------------
Collects domain-specific data:
1. Web scraping
2. PDF extraction
3. Metadata handling

Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""

import os
import requests
from bs4 import BeautifulSoup
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # Load environment variables

DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def scrape_web_page(url: str, output_file: str = "web_data.txt") -> str:
    """Scrape textual content from a webpage."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = "\n".join(paragraphs)

    output_path = DATA_RAW_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] Web page data saved to {output_path}")
    return str(output_path)


def extract_pdf_text(pdf_path: str, output_file: str = "pdf_data.txt") -> str:
    """Extract text from a PDF file."""
    output_path = DATA_RAW_DIR / output_file

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in tqdm(pdf.pages, desc="Extracting PDF text"):
            text += page.extract_text() + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] PDF text saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    urls = [
        "https://en.wikipedia.org/wiki/Charging_station"
    ]
    for url in urls:
        scrape_web_page(url)

    # Example local PDF extraction
    sample_pdf = "data/sample_ev_doc.pdf"
    if os.path.exists(sample_pdf):
        extract_pdf_text(sample_pdf)
    else:
        print("[WARN] Sample PDF not found; skipping PDF extraction.")
