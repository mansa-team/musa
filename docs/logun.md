Logun

mansa's custom financial analysis model trainined/finetuned from the ground up with syntethic data tailored for the brazilain context and many h100's

https://arxiv.org/abs/2506.06335

idea:
    figure out how to download a massive amount of cvm data
    translate the datasets to portuguese using ctranslate2 and vllm
    use diffusiongemma to generate a brazilian tailored dataset based on the translated dataset
    infere 2x more data into the diffusiongemma dataset
    retrain a modernbert-large on this data
    wowww performance

    custom inference mechanism, optimize for latency and apply it to the mansa infraestructure (extended tokenizer with technical data from the brazilian financial context, custom training/data processing pipelines)

    full on training with dapt and posterior stf on the lucas-leme/Sentiments-FinBERT-PT-BR dataset

data composition matrix
    - 50% native structural foundations: raw cvm filings, fatos relevantes, itr/dfp balance sheet logs
    - 25% global market syntatic shift: translated ab30atsiwo/finbert-gpt FinGPT/fingpt-sentiment-train TimKoornstra/financial-tweets-sentiment KalsusEvening/financial-news-headlines
    - 25% synthetically mutated contexts: evol-instruct portuguese data

    - fine-tuning: lucas-leme/Sentiments-FinBERT-PT-BR


develop a benchmark to compare the models on financial data analysis

https://huggingface.co/datasets/lucas-leme/Sentiments-FinBERT-PT-BR
https://huggingface.co/datasets/ab30atsiwo/finbert-gpt
https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train
https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment
https://huggingface.co/datasets/KalsusEvening/financial-news-headlines



https://arxiv.org/abs/2605.00086