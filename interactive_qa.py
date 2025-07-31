from rich.console import Console
from rich.prompt import Prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

console = Console()

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "llama-ev-lora"
device = "cpu"

# Load model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    adapter_loaded = True
except Exception as e:
    console.print(f"[yellow]Warning: Adapter not loaded ({e}). Using base model.[/yellow]")
    model = base_model.to(device)
    adapter_loaded = False

def interactive_chat():
    console.print("[bold yellow]Welcome to EV QA Chat![/bold yellow]")
    console.print("[bold cyan]Type 'exit' to quit.[/bold cyan]\n")
    while True:
        question = Prompt.ask("[bold blue]You[/bold blue]")
        if question.lower() in ["exit", "quit"]:
            console.print("[bold green]Goodbye![/bold green]")
            break
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            inputs = tokenizer(f"Question: {question}", return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=50)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        console.print(f"[bold blue]Q:[/bold blue] {question}")
        console.print(f"[bold green]A:[/bold green] {answer}\n")

if __name__ == "__main__":
    interactive_chat()
