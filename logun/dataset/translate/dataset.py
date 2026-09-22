import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "models"))
os.environ["HF_HOME"] = CACHE_DIR

from datasets import load_dataset
import pandas as pd

dataset = []

# KalsusEvening/financial-news-headlines
dataset.append(load_dataset("KalsusEvening/financial-news-headlines")["train"].to_pandas())

dataset[0] = dataset[0].drop(columns={"sector", "topic", "company"})
dataset[0] = dataset[0].rename(columns={"headline": "source"})

dataset[0]["dataset"] = "KalsusEvening/financial-news-headlines"
dataset[0] = dataset[0][["sentiment", "source", "dataset"]]

# TimKoornstra/financial-tweets-sentiment
dataset.append(load_dataset("TimKoornstra/financial-tweets-sentiment")["train"].to_pandas())
dataset[1] = dataset[1].drop(columns={"url"})
dataset[1] = dataset[1].rename(columns={"tweet": "source"})

dataset[1]["sentiment"] = dataset[1]["sentiment"].map({
    0: "neutral",
    1: "positive",
    2: "negative"
})

dataset[1]["dataset"] = "TimKoornstra/financial-tweets-sentiment"
dataset[1] = dataset[1][["sentiment", "source", "dataset"]]

# FinGPT/fingpt-sentiment-train
dataset.append(load_dataset("FinGPT/fingpt-sentiment-train")["train"].to_pandas())

dataset[2] = dataset[2].drop(columns={"instruction"})
dataset[2] = dataset[2].rename(columns={"output": "sentiment", "input": "source"})

dataset[2]['sentiment'] = dataset[2]['sentiment'].map({
    "strong negative": "negative",
    "strong positive": "positive",

    "mildly negative": "negative",
    "mildly positive": "positive",

    "moderately negative": "negative",
    "moderately positive": "positive",

    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral"
})

dataset[2]["dataset"] = "FinGPT/fingpt-sentiment-train"
dataset[2] = dataset[2][["sentiment", "source", "dataset"]]

# ab30atsiwo/finbert-gpt
dataset.append(load_dataset("ab30atsiwo/finbert-gpt")["train"].to_pandas())
dataset[3] = dataset[3][dataset[3]["source"] != "concat_phrasebank_train"]
dataset[3] = dataset[3].drop(columns={"source", "comb_size", "Unnamed: 0"})
dataset[3] = dataset[3].rename(columns={"sentence": "source", "label": "sentiment"})

dataset[3]["sentiment"] = dataset[3]["sentiment"].map({
    0: "negative",
    1: "neutral",
    2: "positive"
})

dataset[3]["dataset"] = "ab30atsiwo/finbert-gpt"
dataset[3] = dataset[3][["sentiment", "source", "dataset"]]

# concat and dedup
EVAL_PATH = os.path.join(SCRIPT_DIR, "financial_phrase_bank_pt_br.csv")

dataset = pd.concat([dataset[0], dataset[1], dataset[2], dataset[3]], ignore_index=True)

norm = lambda s: s.astype(str).str.lower().str.split().str.join(" ")

eval = set(norm(pd.read_csv(EVAL_PATH)["text"]))
dataset["norm"] = norm(dataset["source"])

dataset = dataset[~dataset["norm"].isin(eval)].drop(columns=["norm"]).reset_index(drop=True)

print(dataset)