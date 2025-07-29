"""
Dataset Generation Module (Open Source)
---------------------------------------
Generates question-answer pairs using a small open LLM (e.g., LLaMA).

Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROCESSED_FILE = "data/processed/processed.csv"
OUTPUT_FILE = "data/processed/qa_dataset.json"

# Load small LLM 
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"


def load_llm_pipeline():
    """Load a local HuggingFace model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def generate_qa_pairs_local(pipe, context: str, n: int = 2) -> list:
    """
    Generate question-answer pairs using local model.
    """
    prompt = (
        f"Generate {n} diverse Question-Answer pairs based on this text:\n"
        f"{context}\n"
        "Format output as JSON list of objects with keys 'question' and 'answer'."
    )

    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    text = outputs[0]["generated_text"]

    # Try to extract JSON (simple heuristic)
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        qa_pairs = json.loads(text[start:end])
    except Exception:
        qa_pairs = []
    return qa_pairs


def generate_dataset():
    """Generate QA dataset using local LLM."""
    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError(f"{PROCESSED_FILE} not found. Run data processing first.")

    df = pd.read_csv(PROCESSED_FILE)
    qa_data = []

    pipe = load_llm_pipeline()

    for text in tqdm(df["text"], desc="Generating Q&A pairs"):
        pairs = generate_qa_pairs_local(pipe, text, n=2)
        for pair in pairs:
            qa_data.append({
                "context": text,
                "question": pair.get("question", ""),
                "answer": pair.get("answer", "")
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2)

    print(f"[INFO] QA dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
