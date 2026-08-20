# Logun: Efficient Portuguese Financial Sentiment Analysis via Domain-Adapted ModernBERT Encoders

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
    
They were selected to cover a range of different approaches on news, which makes the model a better generalist. Since these data is in English, the researcher proposed the usage of LLMs like Qwen 3.5 4B with breakthrough Speculative Decoding technologies like DFlash, with a special version made by inco.ai, called DFlash 2 for mass translation of the labels in a specialized scripts tailored for checkpoints and concurrency on the translation process of the datasets, that are going to be further merged within a data preparation for a final submit for the LoRa pipeline. The LLM setup uses a small context window 4096 tokens to optimize the VRAM headroom for the Speculative Decoding application, it also uses a 4bit quantization to reduce the VRAM footprint further, allowing for high performance throughput in translation tasks.
The scripts will also contain specialized configs for the LLMs for low hallucination profiles with tight temperatures and top-k configs, with a strong and structure system prompt to prevent deviations in outputs. They will also be evaluated using COMET (Unbabel/wmt22-cometkiwi-da at int8) for quality gating

(include benchmarks on tk/s generation and vram usage on different scenarios for qwen with different dflash configs)

As per the results, they are compared in a benchamrk to compare the results in accuracy and latency, it considers the usage of 3 datasets as benchmarks:
    https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation
    https://huggingface.co/datasets/lucas-leme/Sentiments-FinBERT-PT-BR
    https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual (portuguese subset)

These will evaluate the ability of the model in financial scenarios and generalist approaches, the used models are going to be:
    - Qwen 3.5 4B Q4 DFlash2
    - NorBERTo-base
    - NorBERTo-large
    - FinBERT-PT-BR
    - DeB3RTa
    - Logun Base
    - Logun Large

with a script running on each run to verify the latency stats from each model

https://arxiv.org/abs/2605.00086
https://arxiv.org/abs/2506.06335

https://arxiv.org/abs/2602.06036
https://inco.ai/blog/dflash2/

todo:
- [ ] include the cvm scraper specs
- [ ] evaluate the usage of more efficient approaches for DAPT