from pathlib import Path
import os
import yaml
import argparse

from huggingface_hub import list_repo_files, snapshot_download, upload_folder
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
import torch

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

class PushPhaseCallback(TrainerCallback):
    def __init__(self, phase, repo):
        self.phase = phase
        self.repo = repo

    def on_save(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return control
        ckpt = f"checkpoint-{state.global_step}"
        upload_folder(
            repo_id=self.repo,
            folder_path=str(Path(args.output_dir) / ckpt),
            path_in_repo=f"{self.phase}/{ckpt}",
            token=HF_TOKEN,
        )
        return control


class TrueTrainLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            logs["loss_true"] = logs["loss"] / max(1, args.gradient_accumulation_steps)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--resume", nargs="?", const=True, default=False)
args = parser.parse_args()

repository_name = "heitorrosa/logun-base"
model_name = "heitorrosa/logun-base"
checkpoint_name = "logun-base-sft"

config = Path(__file__).resolve().parent / "config.yaml"
config = yaml.safe_load(config.read_text(encoding="utf-8"))

CACHE = Path(__file__).resolve().parent / "models"
CACHE.mkdir(parents=True, exist_ok=True)

DATASET_CACHE = CACHE / "datasets"
DATASET_CACHE.mkdir(parents=True, exist_ok=True)

resume = None
if args.resume:
    if args.resume is True:
        files = list_repo_files(repository_name, token=HF_TOKEN)
        nums = [int(p.split("/")[1].split("-")[1]) for p in files if p.startswith("sft/checkpoint-") and len(p.split("/")) > 2]
        checkpoint = f"checkpoint-{max(nums)}" if nums else None
    else:
        checkpoint = str(args.resume).split("/")[-1]

    target = Path(CACHE / checkpoint_name) / "sft" / checkpoint if checkpoint else None

    if checkpoint and not (target / "trainer_state.json").exists():
        snapshot_download(
            repo_id=repository_name,
            revision="main",
            allow_patterns=[f"sft/{checkpoint}/*"],
            local_dir=Path(CACHE / checkpoint_name),
            token=HF_TOKEN,
        )

    resume = str(target) if checkpoint and target.exists() else None

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE), use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    cache_dir=str(CACHE),
    trust_remote_code=True,
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
)

if hasattr(model, "peft_config"):
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS,
    target_modules=["Wqkv", "Wo", "Wi"],
    modules_to_save=["classifier"]
)

ADAPTER_NAME = "sft"
try: model = get_peft_model(model, lora_config, adapter_name=ADAPTER_NAME)
except TypeError: model = get_peft_model(model, lora_config); ADAPTER_NAME = "default"
try: model.set_adapter(ADAPTER_NAME)
except Exception: pass

dataset = load_dataset("heitorrosa/financial-sentiment-pt", cache_dir=str(DATASET_CACHE))["train"].to_pandas()
dataset = dataset[pd.to_numeric(dataset.get("quality", 0), errors="coerce").fillna(0) >= 0.6]

labels = dataset["sentiment"].astype(str).str.strip().str.lower()

dataset["labels"] = np.select([labels.str.startswith(prefix) for prefix in ("neg", "neu", "pos")], [0, 1, 2], default=-1,)
dataset = dataset.loc[dataset["labels"] >= 0, ["translated", "labels"]].rename(columns={"translated": "text"})

dataset = Dataset.from_pandas(dataset, preserve_index=False).train_test_split(test_size=0.01, seed=config['seed'])

tokenized = dataset.map(
    lambda data: tokenizer(data['text'], truncation=True, max_length=512),
    batched=True, remove_columns=["text"], load_from_cache_file=True
)

training_args = TrainingArguments(
    output_dir=str(CACHE / checkpoint_name / "sft"),

    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=3,

    optim="adamw_torch_fused",
    learning_rate=0.00005, warmup_steps=150,
    weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95,

    fp16=True,
    seed=config['seed'],

    logging_steps=15,
    eval_strategy="steps", eval_steps=150,
    save_steps=150, save_total_limit=3,
    
    push_to_hub=False,

    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

collator = DataCollatorWithPadding(tokenizer)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"], data_collator=collator, compute_metrics=compute_metrics, callbacks=[PushPhaseCallback("sft", repository_name), TrueTrainLossCallback()])

trainer.train(resume_from_checkpoint=resume)
trainer.save_model(str(CACHE / checkpoint_name / "sft"))
upload_folder(
    repo_id=repository_name,
    folder_path=str(CACHE / checkpoint_name / "sft"),
    path_in_repo="sft",
    token=HF_TOKEN,
)
