"""
Test Data Collection Module
---------------------------
Ensures scraping and PDF extraction work as expected.

Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""
import os
from pathlib import Path
import pytest
from src.data_collection import scrape_web_page, extract_pdf_text

DATA_RAW_DIR = Path("data/raw")
TEST_PDF = "data/test_sample.pdf"

@pytest.fixture
def setup_test_pdf():
    # Create a dummy PDF for testing
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a test PDF for EV stations.", ln=True)
    pdf.output(TEST_PDF)
    yield
    os.remove(TEST_PDF)


def test_scrape_web_page():
    url = "https://en.wikipedia.org/wiki/Charging_station"
    output_path = scrape_web_page(url, output_file="test_web.txt")
    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "charging" in content.lower()


def test_extract_pdf_text(setup_test_pdf):
    output_path = extract_pdf_text(TEST_PDF, output_file="test_pdf.txt")
    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "test pdf" in content.lower()
