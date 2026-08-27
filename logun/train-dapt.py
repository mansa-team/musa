from datetime import datetime
from pathlib import Path
import os
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

from dotenv import load_dotenv
load_dotenv()

model_name = 'Itau-Unibanco/NorBERTo-base'
checkpoint_name = f'logun-base-{datetime.now().date()}-mlm'

config = Path(__file__).resolve().parent / "config.yaml"
config = yaml.safe_load(config.read_text(encoding="utf-8"))

CACHE = Path(__file__).resolve().parent / "models"
CACHE.mkdir(parents=True, exist_ok=True)

DATASET_CACHE = CACHE / "datasets"
DATASET_CACHE.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(CACHE), trust_remote_code=True, dtype=torch.bfloat16)

model = get_peft_model(model, LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query","key","value","dense"]
))

dataset = load_dataset("json", data_files="logun/dataset/data/output/corpus-250M.jsonl", cache_dir=str(DATASET_CACHE))["train"].train_test_split(test_size=0.02, seed=config['seed'])
tokenized_dataset = dataset.map(
    lambda data: tokenizer(data['text'], truncation=True, max_length=8192),
    batched=True, remove_columns=["text"], load_from_cache_file=True
)

args = TrainingArguments(
    output_dir=str(CACHE / checkpoint_name),

    per_device_train_batch_size=2,  # 2x8192 ~1.6gb
    per_device_eval_batch_size=4,   # was 8; half for T4 headroom
    gradient_accumulation_steps=16,  # 2x16=32 effective (was 4x8) -> 32x8192=~262ktok/step
    num_train_epochs=1.5,           # 250M sweet spot 1.5 pass, not 3 (overfit KL 0.053)

    optim="adamw_torch_fused",
    learning_rate=0.00005, warmup_steps=500,
    weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95, # adamw_torch_fused
    
    bf16=True,
    seed=config['seed'],

    logging_steps=50,
    eval_strategy="steps", eval_steps=500,
    save_steps=500, save_total_limit=2,

    push_to_hub=True,
    hub_model_id="heitorrosa/logun-base",
    hub_strategy="every_save",
    hub_token=os.getenv("HF_TOKEN"),

    dataloader_pin_memory=True
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["test"], data_collator=collator)

trainer.train(resume_from_checkpoint=False) # use --resume if checkpoint is avaliable
trainer.save(str(CACHE / checkpoint_name))