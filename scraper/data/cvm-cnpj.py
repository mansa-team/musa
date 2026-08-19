import io
import zipfile
from datetime import datetime

import pandas as pd
import requests

CVM_FCA_YEARS = [datetime.now().year, datetime.now().year - 1]
CVM_CAD_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
TICKER_RE = r"^[A-Z0-9]{4}\d{1,2}[A-Z]?$"

# FCA valor_mobiliario: ticker -> CNPJ (official CVM filing, weekly updated)
fca_frames = []
for year in CVM_FCA_YEARS:
    try:
        r = requests.get(f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{year}.zip", timeout=60)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            df = pd.read_csv(zf.open(f"fca_cia_aberta_valor_mobiliario_{year}.csv"), sep=";", encoding="latin1", dtype=str)
        df = df[df["Codigo_Negociacao"].str.match(TICKER_RE, na=False)]
        df = df.rename(columns={"CNPJ_Companhia": "CNPJ", "Codigo_Negociacao": "TICKER"})
        df["CNPJ"] = df["CNPJ"].str.replace(r"\D", "", regex=True).str.zfill(14)
        fca_frames.append(df[["TICKER", "CNPJ"]])
    except:
        pass
fca = pd.concat(fca_frames, ignore_index=True).drop_duplicates(subset="TICKER", keep="first") if fca_frames else pd.DataFrame(columns=["TICKER", "CNPJ"])

# cad_cia_aberta: CNPJ -> CD_CVM enrichment (active registrations first)
cad = pd.read_csv(CVM_CAD_URL, sep=";", encoding="ISO-8859-1", dtype=str)
cad["CNPJ"] = cad["CNPJ_CIA"].str.replace(r"\D", "", regex=True).str.zfill(14)
cad["_rank"] = cad["SIT"].map({"ATIVO": 0, "SUSPENSO(A) - DECISAO ADM": 1, "CANCELADA": 2}).fillna(9)
cad = cad.sort_values(["CNPJ", "_rank", "DT_REG"], ascending=[True, True, False]).drop_duplicates(subset="CNPJ", keep="first")

# tickers from scraper
tickers = pd.DataFrame(requests.get('http://localhost:3200/stocks/fundamental?fields=DY&dates=2026-06-19').json()['data'])[['TICKER', 'NOME']]

# direct FCA merge
results = tickers.merge(fca, on="TICKER", how="left")
results["SOURCE"] = results["CNPJ"].notna().map({True: "FCA", False: None})

# base-ticker fallback for unmatched (share-class variants: 3/4/5/6/11)
unmatched = results[results["CNPJ"].isna()].copy()
if len(unmatched) > 0:
    bases = unmatched["TICKER"].str[:4]
    cands = pd.DataFrame({"TICKER": unmatched["TICKER"].values, "_cand": [b + s for b, s in zip(bases, ["3","4","5","6","11"] * len(bases))]})
    # repeat for all 5 suffixes
    cands = []
    for _, row in unmatched.iterrows():
        b = row["TICKER"][:4]
        for s in ["3", "4", "5", "6", "11"]:
            c = b + s
            if c != row["TICKER"]:
                cands.append({"TICKER": row["TICKER"], "_cand": c})
    cands = pd.DataFrame(cands)
    hits = cands.merge(fca, left_on="_cand", right_on="TICKER", how="left", suffixes=("", "_fca")).dropna(subset=["CNPJ"])
    hits = hits.drop_duplicates(subset="TICKER", keep="first")
    results = results.merge(hits[["TICKER", "CNPJ", "_cand"]], on="TICKER", how="left", suffixes=("", "_fb"))
    results["CNPJ"] = results["CNPJ"].fillna(results["CNPJ_fb"])
    results["SOURCE"] = results["SOURCE"].fillna(results["_cand"].map(lambda x: f"FCA-BASE ({x})" if pd.notna(x) else None))
    results.drop(columns=["CNPJ_fb", "_cand"], inplace=True)

# CNPJ -> CD_CVM
results = results.merge(cad[["CNPJ", "CD_CVM"]], on="CNPJ", how="left")

print(results[["TICKER", "CNPJ", "CD_CVM", "SOURCE"]].to_string(index=False))
print(f"\nmatched: {results['CNPJ'].notna().sum()}/{len(results)} ({100*results['CNPJ'].notna().mean():.1f}%)")
print(results["SOURCE"].value_counts(dropna=False).to_string())


# refactor soon