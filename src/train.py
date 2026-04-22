from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import os

model_name = "gpt2"  # melhor que distilgpt2

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Modelos
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

# Caminho do dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "dataset.jsonl")

# Dataset
dataset = load_dataset("json", data_files=data_path)

# 🔥 Formatar dataset (chat-style)
def format_dataset(example):
    return {
        "prompt": f"Usuário: {example['prompt']}\nAssistente:",
        "chosen": f" {example['chosen']}",
        "rejected": f" {example['rejected']}"
    }

dataset = dataset.map(format_dataset)

# Configuração
training_args = DPOConfig(
    output_dir="../results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    beta=0.1,
    bf16=False,
    fp16=False,
    use_cpu=True
)

# Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer
)

# Treino
trainer.train()

# Salvar
model.save_pretrained("../results/final_model")
tokenizer.save_pretrained("../results/final_model")