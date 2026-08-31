from pathlib import Path
import os
import yaml
import argparse

from huggingface_hub import list_repo_files, snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import torch

from dotenv import load_dotenv
load_dotenv()

class EpochStopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= 0.75:
            control.should_training_stop = True

HF_TOKEN = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser()
parser.add_argument("--resume", nargs="?", const=True, default=False)
args = parser.parse_args()

model_name = 'Itau-Unibanco/NorBERTo-base'
checkpoint_name = f'logun-base-250M'

config = Path(__file__).resolve().parent / "config.yaml"
config = yaml.safe_load(config.read_text(encoding="utf-8"))

CACHE = Path(__file__).resolve().parent / "models"
CACHE.mkdir(parents=True, exist_ok=True)

DATASET_CACHE = CACHE / "datasets"
DATASET_CACHE.mkdir(parents=True, exist_ok=True)

resume = None
if args.resume is not None:
    if args.resume is True:
        files = list_repo_files("heitorrosa/logun-base", token=HF_TOKEN)
        nums = [int(p.split("/")[0].split("-")[1]) for p in files if p.startswith("checkpoint-") and "/" in p]
        checkpoint = f"checkpoint-{max(nums)}" if nums else None
    else:
        checkpoint = str(args.resume).split("/")[-1]

    target = Path(CACHE / checkpoint_name) / checkpoint if checkpoint else None

    if checkpoint and not (target / "trainer_state.json").exists():
        snapshot_download(
            repo_id="heitorrosa/logun-base",
            revision="main",
            allow_patterns=[f"{checkpoint}/*"],
            local_dir=Path(CACHE / checkpoint_name),
            token=HF_TOKEN,
        )

    resume = str(target) if checkpoint and target.exists() else None
    
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(CACHE), trust_remote_code=True, dtype=torch.float16)

model = get_peft_model(model, LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["Wqkv", "Wo", "Wi"]
))

dataset = load_dataset("json", data_files="logun/dataset/data/output/corpus-250M.jsonl", cache_dir=str(DATASET_CACHE))["train"].train_test_split(test_size=0.01, seed=config['seed'])

tokenized_dataset = dataset.map(
    lambda data: tokenizer(data['text'], truncation=True, max_length=8192),
    batched=True, remove_columns=["text"], load_from_cache_file=True
)

training_args = TrainingArguments(
    output_dir=str(CACHE / checkpoint_name),

    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=1.5,

    optim="adamw_torch_fused",
    learning_rate=0.00005, warmup_steps=500,
    weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95,

    fp16=True,
    seed=config['seed'],

    logging_steps=50,
    eval_strategy="steps", eval_steps=500,
    save_steps=500, save_total_limit=2,

    push_to_hub=True,
    hub_model_id="heitorrosa/logun-base",
    hub_strategy="all_checkpoints",
    hub_token=HF_TOKEN,

    dataloader_pin_memory=True,
    gradient_checkpointing=True,
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["test"], data_collator=collator, callbacks=[EpochStopCallback()])

trainer.train(resume_from_checkpoint=resume)
trainer.save_model(str(CACHE / checkpoint_name))
