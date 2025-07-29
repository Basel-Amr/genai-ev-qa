# genai-ev-qa
 End-to-end pipeline for domain-specific fine-tuning and deployment of small language models. Automates data collection, preprocessing, model fine-tuning, evaluation, and production-ready API serving.

 ## Introduction
LLMForge provides an automated workflow to:
- Collect domain-specific data (e.g., PDFs, web sources)
- Fine-tune small language models (like LLaMA-3-7B)
- Evaluate and benchmark model performance
- Deploy the fine-tuned model as a scalable production API

This repository is designed for data scientists, ML engineers, and researchers who want to quickly adapt LLMs to their custom use cases.

## Features
- **Automated Data Collection:** Scrape & extract structured text from PDFs and web sources.
- **Data Preprocessing Pipeline:** Cleaning, filtering, and tokenization.
- **Fine-Tuning:** Easily fine-tune small LLMs with domain-specific data.
- **Evaluation & Benchmarking:** Automated tests using standard metrics and custom QA tasks.
- **Production Deployment:** REST API using FastAPI with Docker and CI/CD.

## Tech Stack
- Python 3.10+
- Hugging Face Transformers & PEFT
- PyTorch
- FastAPI (for API Serving)
- Docker & Docker Compose
- Weights & Biases (for experiment tracking)

## Pipeline Overview
1. **Data Collection** → 2. **Preprocessing** → 3. **Fine-Tuning** → 4. **Evaluation** → 5. **Deployment**

## Quick Start
```bash
# Clone the repo
https://github.com/Basel-Amr/genai-ev-qa.git
cd genai-ev-qa

# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train.py --config configs/finetune.yaml

# Serve model
uvicorn app.main:app --reload


