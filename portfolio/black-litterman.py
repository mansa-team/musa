import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time

from sklearn.pipeline import Pipeline
from skfolio.pre_selection import DropCorrelated, DropZeroVariance

start_time = time.time() 
np.random.seed(42)

N_STOCKS = 18

try:
    selic_monthly = pd.DataFrame(requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados?formato=json").json()).rename(columns={"data": "DATE", "valor": "MONTHLY"})
    selic_anually = pd.DataFrame(requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json").json()).rename(columns={"data": "DATE", "valor": "ANNUAL"})

    selic = pd.merge(selic_monthly, selic_anually, on='DATE', how='inner')
    selic['DATE'] = pd.to_datetime(selic['DATE'], dayfirst=True)
    selic = selic.set_index('DATE')

    print(selic.tail(5))
except Exception as e:
    print(f"error: {e}")