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
    https://huggingface.co/datasets/ab30atsiwo/finbert-gpt
    https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train
    https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment
    https://huggingface.co/datasets/KalsusEvening/financial-news-headlines
    
They were selected to cover a range of different approaches on news, which makes the model a better generalist. Since these data is in English, the researcher proposed the usage of LLMs like Qwen 3.5 4B with breakthrough Speculative Decoding technologies like dspark/mtp for mass translation of the labels in a specialized scripts tailored for checkpoints and concurrency on the translation process of the datasets, that are going to be further merged within a data preparation for a final submit for the LoRa pipeline. The LLM setup uses a small context window 4096 tokens to optimize the VRAM headroom for the Speculative Decoding application, it also uses a 4bit quantization to reduce the VRAM footprint further, allowing for high performance throughput in translation tasks.
The scripts will also contain specialized configs for the LLMs for low hallucination profiles with tight temperatures and top-k configs, with a strong and structure system prompt to prevent deviations in outputs. They will also be evaluated using COMET (Unbabel/wmt22-cometkiwi-da at int8) for quality gating

(include benchmarks on tk/s generation and vram usage on different scenarios for qwen with different dspark/mtp configs)

As per the results, they are compared in a benchamrk to compare the results in accuracy and latency, it considers the usage of 3 datasets as benchmarks:
    https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation
    https://huggingface.co/datasets/lucas-leme/Sentiments-FinBERT-PT-BR
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

the corpus split script uses Data Selection via Importance Resampling, formalized by Xie et al. 2023, in which hashed 1+2-gram features are put into 10k buckets with Laplace smoothing (alpha 1.0) to get gamma (target) and nu (source) and then each chunk just gets log w = sum cnt * log(gamma/nu) and we sort top until 250M, which is way better than a dumb FINANCIAL_TERMS list cause that misses paraphrases like oferta pública vs alienação de controle and is super brittle, also beats classifier stuff like fastText/BERT that need training and overfit and embedding centroid that needs gpu and has hubness and perplexity/CED that collapses to mode — DSIR is just split+zlib.crc32 + Counter so its minutes on cpu even for 924k, works for pt-br without lang model, generative so it dont overfit like discriminative (-0.6% worse in Xie), bigram gets +0.26 over unigram and the KL reduction actually predicts F1 with r=0.82 so we know before training its good.

for the sft pipeline, the dataset translation will include an api proxy with google colab's free tier for shared inference speeds on qwen 3.5 4b with the 1660super setup. for reproducibility purpouse, the inference engine will be wrapped around a docker container, allowing for easy replication across environments. the translation script will use both serving endpoints for concurrency stability, allowing the script to serve multiple translations per iteration because of the high throughput created by the fusion of a small model + dspark/mtp.

the inference engine will prob run on some dspark/mtp compatible like llama.cpp, the researcher also consider the benchmark of:
- qwen 3.5 4b mtp from alibaba
- lfm 2.5 8b a1b dspark from liquidai

to evaluate the token throughput performance compared to its accuracy in some benchmark focused on translation performance, its hypothesised that the lfm 8b a1b will heavily outperform qwen 3.5 4b mtp because of its smaller footprint thanks to moe that only activates 1b parameters and the 320m param drafter that it has for dspark speculative decoding and since its trained on 38T tokens and oficially supports portuguse as one of its languages, it will have a good enough performance, comapred to qwen 3.5 that was pretrained on 36T tokens and is meant for a more broader language population.

# dapt
https://sol.sbc.org.br/index.php/bwaif/article/view/24960
https://arxiv.org/abs/2004.10964
https://arxiv.org/abs/2302.03169
https://arxiv.org/abs/2512.12384

https://github.com/pytorch/pytorch/issues/121957

# translation and inference
https://arxiv.org/abs/2506.06335
https://arxiv.org/abs/2602.06036
https://arxiv.org/abs/2607.05147

# base models
https://arxiv.org/pdf/2511.23404
https://arxiv.org/pdf/2505.09388

https://arxiv.org/abs/2605.00086
https://arxiv.org/pdf/2606.22722 (possible alternative for the non cc NorBERTo)

todo:
- [ ] include the cvm scraper specs
- [ ] evaluate the usage of more efficient approaches for DAPT
- [ ] replace "we are going to use x" to "Hypothesis -> Experiment -> Measurement -> Expected interpretation"

specs:
1660super 6gb 192gb/s
4x8gb 25gb/s
