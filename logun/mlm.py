from datetime import datetime
from pathlib import Path
import yaml

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

model_name = 'Itau-Unibanco/NorBERTo-base'
checkpoint_name = f'logun-base {datetime.now().date()}-mlm'

config = Path(__file__).resolve().parent / "config.yaml"
config = yaml.safe_load(config.read_text(encoding="utf-8"))

CACHE = Path(__file__).resolve().parent / "models"
CACHE.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(CACHE), trust_remote_code=True)

model = get_peft_model(model, LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query","key","value","dense"]
))

dataset = load_dataset("json", data_files="logun/dataset/data/output/corpus-250M.jsonl")["train"].train_test_split(test_size=0.02, seed=config['seed'])
tokenized_dataset = dataset.map(
    lambda data: tokenizer(data['text'], truncation=True, max_length=8192),
    batched=True, remove_columns=["text"]
)

args = TrainingArguments(
    output_dir=str(CACHE / "logun-base-250M"),

    per_device_train_batch_size=4,  # 4x8192 tokens ~3.2gb
    per_device_eval_batch_size=8,   # eval has no gradient, so 2x
    gradient_accumulation_steps=8,  # 4x8=32 -> 32x8192=~262ktok/step
    num_train_epochs=1.5,           # 250M sweet spot 1.5 pass, not 3 (overfit KL 0.053)
    learning_rate=0.00005, warmup_ratio=0.06,
    weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95, # adamw_torch_fused
    fp16=True,
    seed=config['seed'],
    logging_steps=50,
    eval_strategy="steps", eval_steps=500,
    save_steps=500, save_total_limit=2,
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["test"], data_collator=collator)

trainer.train()
trainer.save(str(CACHE / checkpoint_name))

# about 2 days of training in a 1660super + 32gb