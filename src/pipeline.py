"""
End-to-End LLM Fine-Tuning Pipeline
-----------------------------------
Steps:
1. Generate QA dataset (handles long context with chunking)
2. Fine-tune TinyLlama using LoRA
3. Evaluate using SQuAD metrics, BLEU, and ROUGE
4. Save & register model for Hugging Face deployment

Framework: Prefect
Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""

import os
import json
import random
import re
import string
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from prefect import flow, task
from rich.console import Console
from rich.table import Table

console = Console()

# --- Config ---
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "llama-ev-lora"
DATA_FILE = "data/processed/qa_dataset.json"
MAX_INPUT_LENGTH = 512
EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 4
SEED = 42


# -------------------------
# Utility Functions
# -------------------------
def normalize_text(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def squad_metrics(predictions, references):
    exact_matches, f1s = [], []
    for pred, ref in zip(predictions, references):
        pred_tokens, ref_tokens = normalize_text(pred).split(), normalize_text(ref).split()
        exact_matches.append(int(pred_tokens == ref_tokens))
        common = set(pred_tokens) & set(ref_tokens)
        num_same = len(common)
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1 = int(pred_tokens == ref_tokens)
        elif num_same == 0:
            f1 = 0
        else:
            precision, recall = num_same / len(pred_tokens), num_same / len(ref_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        f1s.append(f1)
    return 100 * sum(exact_matches) / len(exact_matches), 100 * sum(f1s) / len(f1s)


# -------------------------
# Prefect Tasks
# -------------------------
@task
def load_dataset():
    console.log(f"Loading dataset from {DATA_FILE}")
    with open(DATA_FILE) as f: data = json.load(f)
    random.seed(SEED)
    random.shuffle(data)
    split = int(0.9 * len(data))
    return data[:split], data[split:]

@task
def tokenize_data(train_data, val_data):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        text = f"Question: {example['question']}\nAnswer: {example['answer']}"
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=256)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_dataset = Dataset.from_list(train_data).map(preprocess)
    val_dataset = Dataset.from_list(val_data).map(preprocess)
    return tokenizer, train_dataset, val_dataset

@task
def train_model(tokenizer, train_dataset, val_dataset):
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        output_dir=OUTPUT_DIR,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    def collate_fn(batch):
        return {"input_ids":torch.tensor([x["input_ids"] for x in batch]),
                "attention_mask":torch.tensor([x["attention_mask"] for x in batch]),
                "labels":torch.tensor([x["labels"] for x in batch])}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        preds_text = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        labels_text = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
        exact_match, f1 = squad_metrics(preds_text, labels_text)
        bleu = corpus_bleu(preds_text, [labels_text]).score
        scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
        rouge1 = np.mean([scorer.score(r,p)["rouge1"].fmeasure for r,p in zip(labels_text,preds_text)])
        rougel = np.mean([scorer.score(r,p)["rougeL"].fmeasure for r,p in zip(labels_text,preds_text)])
        return {"exact_match":exact_match,"f1":f1,"bleu":bleu,"rouge1":rouge1,"rougeL":rougel}

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer,
                      data_collator=collate_fn,
                      compute_metrics=compute_metrics)
    trainer.train()

    console.log("[green]Training complete[/green]")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return trainer

@task
def evaluate_model(trainer, val_dataset):
    predictions = trainer.predict(val_dataset)
    table = Table(title="Final Evaluation Metrics (SQuAD-style)")
    for k, v in predictions.metrics.items():
        table.add_row(k, f"{v:.2f}")
    console.print(table)

@task
def register_model():
    console.log(f"Model artifacts saved to {OUTPUT_DIR}. Upload to Hugging Face Space manually or via API.")
    return True

# -------------------------
# Prefect Flow
# -------------------------
@flow
def full_pipeline():
    console.log(f"Pipeline started at {datetime.now()}")
    train_data, val_data = load_dataset()
    tokenizer, train_dataset, val_dataset = tokenize_data(train_data, val_data)
    trainer = train_model(tokenizer, train_dataset, val_dataset)
    evaluate_model(trainer, val_dataset)
    register_model()
    console.log("[bold green]Pipeline complete[/bold green]")


if __name__ == "__main__":
    full_pipeline()
