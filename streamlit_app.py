import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Load environment variables ---
API_TOKEN = os.getenv("API_TOKEN", "")  # read from secret
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODEL_PATH = os.getenv("MODEL_PATH", "llama-ev-lora")

# --- Load model ---
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
try:
    model = PeftModel.from_pretrained(base_model, MODEL_PATH).to(device)
    adapter_loaded = True
except Exception as e:
    model = base_model.to(device)
    adapter_loaded = False

# --- Streamlit UI ---
st.title("EV QA Model (LoRA Fine-Tuned)")

user_question = st.text_input("Ask a question about EV charging:")
if st.button("Get Answer") and user_question:
    with st.spinner("Thinking..."):
        inputs = tokenizer(f"Question: {user_question}", return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not adapter_loaded:
            answer += "\n\n[NOTE] Using base TinyLlama (adapter not loaded)"
    st.success(answer)

st.caption("Powered by TinyLlama + LoRA adapter")
