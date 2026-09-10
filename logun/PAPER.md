# Logun: Efficient Domain Adaptation and Parameter-Efficient Fine-Tuning for Portuguese Financial Language Understanding

Logun is an approach focusing on the development of finetuning methodologies for Financial Sentiment Analysis in Portuguese based on a pretrained encoding model.
For this approach, the chosen baseline model by the researcher to start is NorBERTo, based on the ModernBERT architecture and trained on 331 billion tokens composed by a different span of datasets, including the AuroraPT dataset by Instituto de Ciência e Tecnologia Itaú, its currently the most efficient and best Portuguese inclined encoding model avaliable according to benchmarks and it allows for the tuning of 2 different Logun models, a Base (150M parameters) and a Large (395M parameters) version, both focused on different efficiency profiles, which could be helpful depending on the task in which each model is deployed, either through non concurrent classification tasks (eg. via an API) or via mass concurrent classification tasks (eg. a scraper running on different news).

ASSIN 2
| Model | Size | Entailment F1 | Similarity (Pearson) |

|:------|:----:|:-------------:|:--------------------:|

| Albertina PT-BR | base | 0.874 | 0.826 |

| BERTimbau | base | 0.883 | 0.836 |

| BERTimbau | large | 0.889 | 0.852 |

| mmBERT | base | 0.896 | 0.821 |

| NorBERTo | base | 0.890 | 0.736 |

| NorBERTo (cross encoder) | large | 0.903 | 0.766 |

| NorBERTo (sequence classification) | large | 0.904 | — |


PLUE
| Model | Size | MRPC | RTE | WNLI |

|:------|:----:|:----:|:---:|:----:|

| Albertina PT-BR | base | 0.878 | 0.646 | 0.549 |

| BERTimbau | large | 0.887 | 0.755 | 0.563 |

| mmBERT | base | 0.905 | 0.758 | 0.563 |

| NorBERTo | base | 0.893 | 0.722 | 0.577 |

| NorBERTo | large | 0.919 | 0.769 | 0.577 |

The chosen finetuning methodology by the researcher is LoRa, it's choice is mainly because of the hardware limitations of the researcher's environment, given the amount of data that is going to be ingested by the model during the finetuning phase, in which, is going to be detailed further in other sections of this paper, the researcher also acknowledges that other approaches like SFT could possibly output better results for the following benchmarks and there's a intent on the researchers part to iterate this paper further in other model checkpoints with different approaches, more data and compute capabilities

For a more in depth domain on financial language in Portuguese corpus text, the research also includes the usage of extended pre-training on CVM financial data for DAPT, this approach allows for the ModernBERT model to better adapt its weights for a posterior LoRa Supervised Fine-Tuning and proper exposure of these tokens in the embedding space, which is going to improve the accuracy of the model in classification tasks in which financial tailored speech is included.

The data corpus for this fine tuning approach consists of a data from a range of datasets that were curated using LLMs for accurate translations and iterations on a Portuguese corpus. The used datasets are:
    https://huggingface.co/datasets/ab30atsiwo/finbert-gpt (phrasebank ones dropped)
    https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train
    https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment
    https://huggingface.co/datasets/KalsusEvening/financial-news-headlines
    
They were selected to cover a range of different approaches on news, which makes the model a better generalist. Since these data is in English, the researcher proposed the usage of LLMs like Qwen 3.5 4B with breakthrough Speculative Decoding technologies like dspark/mtp for mass translation of the labels in a specialized scripts tailored for checkpoints and concurrency on the translation process of the datasets, that are going to be further merged within a data preparation for a final submit for the LoRa pipeline. The LLM setup uses a small context window 4096 tokens to optimize the VRAM headroom for the Speculative Decoding application, it also uses a 4bit quantization to reduce the VRAM footprint further, allowing for high performance throughput in translation tasks.
The scripts will also contain specialized configs for the LLMs for low hallucination profiles with tight temperatures and top-k configs, with a strong and structure system prompt to prevent deviations in outputs. They will also be evaluated using COMET (Unbabel/wmt22-cometkiwi-da at int8) for quality gating

[include benchmarks on tk/s generation and vram usage on different scenarios for qwen with different dspark/mtp configs]

As per the results, they are compared in a benchamrk to compare the results in accuracy and latency, it considers the usage of 3 datasets as benchmarks:
    https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation
    https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual (portuguese subset)

These will evaluate the ability of the model in
    - NorBERTo-base
    - NorBERTo-large
    - FinBERT-PT-BR
    - DeB3RTa
    - Logun Base
    - Logun Large

with a script running on each run to verify the latency stats from each model

for size optimization matters, the researcher extracted the cvm fillings from 2003 - 2020, totalling about 122gb, which, when passed through the prepare.py for dedup and chunking for the final corpus totalled about 6.75gb of corpus data. Ill later on run the scraper on 2021 - 2026 data for a improved corpus, right now, the corpus contains 2.35B tokens.

according to dapt papers, the sweetspot for tokens on dapt is about 250M, which will be used as input for the split.py script, this sums to about 2days in continuous training in the 1660super at 3 epochs. The dapt script will use MLM on LoRa for efficient VRAM usage for the newly trained model.

We select DAPT data with DSIR (Xie et al. 2023): hashed uni+bigram features in 10k buckets estimate target vs. source distributions, each chunk is scored by its importance weight, and we keep the top chunks up to 250M tokens. Unlike keyword lists, it catches paraphrases; unlike classifier or embedding filters, it needs no training, no GPU, and no language model, minutes on CPU. Selection quality is known before training: the KL cut (0.0769 -> 0.0534) predicts downstream F1 with r = 0.82.

[categories split plot from the dsir seelction over the dataset]

"kl_divergence": {
  "pre": 0.0769,
  "post": 0.0534,
  "reduction": 0.0234,
  "ratio": 0.305,
  "gate_pass": true
},
"gates": {
  "duplicate_ratio": 0.0,
  "duplicate_pass": true,
  "hoje_ratio": 0.0269,
  "hoje_pass": true,
  "kl_gate_pass": true,
  "expected_threshold": 375977.6,
  "r_predictive": 0.82
},

for the sft pipeline, the dataset translation will include an api proxy with google colab's free tier for shared inference speeds on qwen 3.5 4b with the 1660super setup. for reproducibility purpouse, the inference engine will be wrapped around a docker container, allowing for easy replication across environments. the translation script will use both serving endpoints for concurrency stability, allowing the script to serve multiple translations per iteration because of the high throughput created by the fusion of a small model + dspark/mtp.

the inference engine will prob run on some dspark/mtp compatible like llama.cpp, the researcher also consider the benchmark of:
- qwen 3.5 4b mtp from alibaba
- lfm 2.5 8b a1b dspark from liquidai

to evaluate the token throughput performance compared to its accuracy in some benchmark focused on translation performance, its hypothesised that the lfm 8b a1b will heavily outperform qwen 3.5 4b mtp because of its smaller footprint thanks to moe that only activates 1b parameters and the 320m param drafter that it has for dspark speculative decoding and since its trained on 38T tokens and oficially supports portuguse as one of its languages, it will have a good enough performance, comapred to qwen 3.5 that was pretrained on 36T tokens and is meant for a more broader language population.

the 1660super is not good enough for this kind of training because of its emulated fp16 (~0.5tflops) vs fp32 (~10tflops), the lack of some kernels for performance like FlashAttention and my limited RAM bandwidth that could only be solved by upgrading the researchers GPU to some that supports Tensor Cores and proper fp16 or by using a cloud GPU like Google Colab's T4, which is free and will be used via colab-ssh for training alongside with checkpoint abusing for continuous runtime. The 1660 Super managed to make one checkpoint logun-base-2026-08-26-mlm-checkpoint-500 ({'loss': '0.6489', 'grad_norm': '0.08104', 'learning_rate': '4.99e-05', 'epoch': '0.08685'}) but it took about 13hours to train, which would make the project not viable, taking about ~7days just for the dapt on the base model (150m params). I tried running autoresearch with Deepseek v4 Flash 0731 for possible inference improvements on training but it wasnt really effective for the cost $1.00 in about 100M tokens it generated.

to make use of the colab gpu, i needed to setup a huggingface repo and i had to setup some args to the trainerArguments in my loop. the t4 setup takes about 1:30h to make a single checkpoint, 10x faster than the 1660super setup, the jupyter notebook was designed to allow for quick and easy resume if a t4 insteance hits its quota, allowing me to train the model easily using any jupyter compatible provider (kaggle, colab, etc.)

the final checkpoint was set at checkpoint-8724, resulting in a 0.4919 eval loss, the researcher decided to merge the checkpoint-8500 for its marginally better performance at 0.4874 eval loss at epoch 1.4615

[loss curve plot on the dapt model]

the dapt has also been shown to be effective, helping the newly trained model to understand cvm language better as per a simple holdout mlm benchmark shows

{
  "base": {
    "loss": 1.1733974539316618,
    "ppl": 3.232957601547241,
    "acc": 0.747016706443914,
    "nMasked": 3771
  },
  "dapt": {
    "loss": 0.8408773816548861,
    "ppl": 2.3184001445770264,
    "acc": 0.8069477592150623,
    "nMasked": 3771
  }
}

leakege checks also report a "0/100 held-out docs occur verbatim in the training pool (0.00%)", indicating valid results

after the dapt on the initial refernece model, the researcher started the setup for the benchmarks on the llms for the mass translation operations, it's mainly going to use llama.cpp as the inference engine for easy reproduction of research and to allow the usage of gguf q4 models, the downloaded models are:
- https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF
- https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF
- https://huggingface.co/unsloth/Qwen3.5-4B-GGUF

its going to have the translation engine decoupled from the inference, making the researcher able to either use the local setup for translation and a external environment with google colab for that. in which, the user would only have to place in the config the urls for the apis, making everything easier and more manageable, the script will also contain a checkpoint system, so the researcher can resume the mass translation at any time.

with the newly release of minicpm5 2b from openbmb on september 7th, its also going to be included in the benchmarks for the llms considering its incridible performance at the Artificial Analysis Intelligence Index v4.2 benchmark, outperforming all other candidates, with 14 points when compared to the 11 and 7 points from alibaba and liquidai respectably, another thing that pops up when evaluating this model is the -12 on the AA-Omniscience Index, where it outperforms both models considerably in terms of hallucination, which could be a positive point for the scenario of mass translation, where accuracy, even at temperature 0.0 is essential.
- https://huggingface.co/openbmb/MiniCPM5-2B-GGUF
- https://huggingface.co/openbmb/MiniCPM5-2B-DSpark

[table showing the benchmarks from minicpm5 against qwen 3.5 4b and lfm 2.5 8b a1b]

# dapt
https://sol.sbc.org.br/index.php/bwaif/article/view/24960 (finbert ptbr, 2023)
https://arxiv.org/abs/2004.10964 (dont stop pretraining, 2020)
https://arxiv.org/abs/2302.03169 (dsir, 2023)
https://arxiv.org/abs/2512.12384 (scaling laws from continued pretraining, 2025)

https://github.com/pytorch/pytorch/issues/121957 (1660super fp16 issue, )

# translation and inference
https://arxiv.org/abs/2506.06335 (finbert2, 2025)
https://arxiv.org/abs/2603.22186 (two stage llm translation, 2026)
https://arxiv.org/abs/2602.06036 (dflash, 2026)
https://arxiv.org/abs/2607.05147 (dspark, 2026)
https://artificialanalysis.ai/models?models=qwen3-5-4b-non-reasoning%2Clfm2-5-8b-a1b%2Cminicpm5-2b#speed (artificial analysis, 2026)


# base models
https://arxiv.org/abs/2511.23404 (lfm2 report, 2025)
https://arxiv.org/abs/2505.09388 (qwen3 report, 2025)
https://arxiv.org/abs/2506.07900 (minicpm4 report, 2025)

https://arxiv.org/abs/2605.00086 (norberto, 2026)
https://arxiv.org/abs/2606.22722 (moberto, 2026)

# core
https://arxiv.org/abs/1810.04805 (BERT, 2018)
https://arxiv.org/abs/1907.11692 (RoBERTa, 2019)
https://arxiv.org/abs/2106.09685 (LoRA, 2021)
https://arxiv.org/abs/2305.14314 (QLoRA, 2023)

https://arxiv.org/abs/1908.10063 (FinBERT, 2019)
https://aclanthology.org/W14-1405/ (Financial PhraseBank, 2014)

https://arxiv.org/abs/2203.15556 (Chinchilla / compute-optimal training, 2022)
https://arxiv.org/abs/2001.08361 (Scaling Laws for Neural Language Models, 2020)

https://www.mdpi.com/2078-2489/11/10/484 (Portuguese NLI / STS benchmarking, includes ASSIN2, 2020)

https://arxiv.org/abs/2211.17192 (Speculative Decoding, 2022)
https://aclanthology.org/2022.wmt-1.60/ (COMET / MT quality estimation, 2022)
https://aclanthology.org/P02-1040/ (BLEU, 2002)

https://arxiv.org/abs/2001.08361 (Scaling Laws, 2020)

todo:
- [ ] include the cvm scraper specs
- [ ] include graphs for eda
- [ ] replace "we are going to use x" to "Hypothesis -> Experiment -> Measurement -> Expected interpretation"

specs:
1660super 6gb 192gb/s
4x8gb 25gb/s

related urls:
- https://huggingface.co/heitorrosa/logun-base
- https://huggingface.co/datasets/heitorrosa/cvm-corpus