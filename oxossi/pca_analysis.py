import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from datetime import datetime
from scipy.signal import detrend
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

nowYear = datetime.now().year
yearsRange = [str(y) for y in range(nowYear - 10, nowYear)]
baseDir = os.path.dirname(os.path.abspath(__file__))
apiBase = "http://localhost:3200"
fundFields = "XANGO INVESTING SCORE"
histFields = "LUCRO LIQUIDO,RECEITA LIQUIDA,MARGEM LIQUIDA"
channels = [c.strip() for c in histFields.split(",")]
shrinkVal = 0.3
nBoot = 500
rng = np.random.default_rng(42)

selicRaw = pd.DataFrame(requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json").json())
selicRaw["valor"] = selicRaw["valor"].astype(float)
selicRaw["data"] = pd.to_datetime(selicRaw["data"], dayfirst=True)
selicFrame = selicRaw.set_index("data").resample("YE").mean()
selicFrame.index = selicFrame.index.year.astype(str)
selicFrame = selicFrame.reindex(yearsRange)
selicVals = selicFrame["valor"].values.astype(float)

fundResp = requests.get(f"{apiBase}/stocks/fundamental", params={"fields": fundFields, "dates": "2026-06-29"}).json()["data"]
fundFrame = pd.DataFrame(fundResp)
selTickers = fundFrame[(fundFrame[fundFields] > 60) & (fundFrame["TICKER"].str.endswith("3"))]["TICKER"].tolist()
tickersParam = ",".join(fundFrame[fundFrame["TICKER"].str.endswith("3")]["TICKER"].tolist())
histResp = requests.get(f"{apiBase}/stocks/historical", params={"fields": histFields, "search": tickersParam}).json()["data"]
histFrame = pd.DataFrame(histResp).drop(columns=["NOME"], errors="ignore").set_index("TICKER")

def channelMatrix(frame, chan, yearLabels):
    cols = [c for c in frame.columns if c.startswith(chan)]
    yearOf = {c: "".join([ch for ch in c if ch.isdigit()])[-4:] for c in cols}
    keep = {c: y for c, y in yearOf.items() if y in yearLabels}
    sub = frame[list(keep.keys())].rename(columns=keep)
    sub = sub[yearLabels].apply(pd.to_numeric, errors="coerce").dropna()
    return sub

chanMats = [channelMatrix(histFrame, c, yearsRange) for c in channels]
commonTickers = sorted(set(selTickers) & set.intersection(*[set(m.index) for m in chanMats]))
chanVals = [m.loc[commonTickers].values.astype(float) for m in chanMats]
nStocks = len(commonTickers)
nYears = len(yearsRange)

def pooledEpsilon(logVals, selicVec):
    detr = np.apply_along_axis(detrend, 1, logVals)
    mFac = detr.mean(axis=0)
    sStd = (selicVec - selicVec.mean()) / selicVec.std() * detr.std()
    regDf = pd.DataFrame({"y": detr.flatten(), "M": np.tile(mFac, nStocks), "S": np.tile(sStd.flatten(), nStocks)})
    fitRes = smf.ols("y ~ M + S", data=regDf).fit()
    eps = detr - (fitRes.params["Intercept"] + fitRes.params["M"] * mFac + fitRes.params["S"] * sStd.flatten())
    return eps

epsList = [pooledEpsilon(np.log1p(np.maximum(v, 0)), selicVals) for v in chanVals]
epsStack = np.hstack(epsList)

def latentCorr(scores):
    zScores = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-12)
    spearMat, _ = spearmanr(zScores.T)
    spearMat = np.atleast_2d(spearMat)
    np.fill_diagonal(spearMat, 1.0)
    spearMat = np.nan_to_num(spearMat, nan=0.0)
    distMat = squareform(pdist(scores, metric="euclidean"))
    eucCorr = 1 - 2 * (distMat / (distMat.max() + 1e-12))
    latMat = (spearMat + eucCorr) / 2
    np.fill_diagonal(latMat, 1.0)
    return latMat

nComp = min(3, nStocks, epsStack.shape[1])
scaler = StandardScaler()
scaledStack = scaler.fit_transform(epsStack)
pcaModel = PCA(n_components=nComp)
scores = pcaModel.fit_transform(scaledStack)
latMat = latentCorr(scores)
baseCorr, _ = spearmanr(epsList[0].T)
np.fill_diagonal(baseCorr, 1.0)
shrunkBase = (1 - shrinkVal) * baseCorr + shrinkVal * np.eye(nStocks)

bootMats = np.zeros((nBoot, nStocks, nStocks))
for b in range(nBoot):
    yearIdx = rng.choice(nYears, nYears, replace=True)
    bootStack = np.hstack([e[:, yearIdx] for e in epsList])
    bootScaled = (bootStack - bootStack.mean(axis=0)) / (bootStack.std(axis=0) + 1e-12)
    bootScores = PCA(n_components=nComp).fit_transform(bootScaled)
    bootMats[b] = latentCorr(bootScores)
bootZ = np.arctanh(np.clip(bootMats, -0.999, 0.999))
ciWidth = np.tanh(np.percentile(bootZ, 97.5, axis=0)) - np.tanh(np.percentile(bootZ, 2.5, axis=0))
np.fill_diagonal(ciWidth, 0)

def triStats(mat):
    tri = mat[np.triu_indices(nStocks, k=1)]
    return np.mean(tri), np.median(tri), tri

latMean, latMed, latTri = triStats(latMat)
rawMean, rawMed, rawTri = triStats(baseCorr)
shrMean, shrMed, shrTri = triStats(shrunkBase)
ciMean = ciWidth[np.triu_indices(nStocks, k=1)].mean()

fig, axes = plt.subplots(1, 3, figsize=(30, 10))
for ax, mat, ttl, cmap, ctr in [
    (axes[0], pd.DataFrame(latMat, index=commonTickers, columns=commonTickers), f"Latent corr (PCA k={nComp})", "coolwarm", 0),
    (axes[1], pd.DataFrame(baseCorr, index=commonTickers, columns=commonTickers), "Baseline eps corr (LL)", "coolwarm", 0),
    (axes[2], pd.DataFrame(ciWidth, index=commonTickers, columns=commonTickers), "Latent 95% CI width (time-boot)", "YlOrRd", None),
]:
    sns.heatmap(mat, cmap=cmap, center=ctr, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"label": "rho" if ctr == 0 else "CI width"})
    ax.set_title(ttl)
plt.tight_layout()
plt.savefig(os.path.join(baseDir, "pca_latent_corr.png"), dpi=150)
plt.close()

loadings = pd.DataFrame(pcaModel.components_.T, index=[f"{c}@{y}" for c in channels for y in yearsRange],
                        columns=[f"PC{i+1}" for i in range(nComp)])
print(f"stocks={nStocks} years={nYears} channels={len(channels)} k={nComp}")
print(f"explained={np.round(pcaModel.explained_variance_ratio_, 3).tolist()} total={pcaModel.explained_variance_ratio_.sum():.3f}")
print(loadings.round(3).to_string())
print(f"latent mean={latMean:.3f} med={latMed:.3f} | base mean={rawMean:.3f} med={rawMed:.3f} | shrunk mean={shrMean:.3f} med={shrMed:.3f} | delta={latMean - rawMean:+.3f}")
print(f"CI width mean={ciMean:.3f}")
pairs = [(commonTickers[i], commonTickers[j], latMat[i, j]) for i in range(nStocks) for j in range(i + 1, nStocks)]
pairs.sort(key=lambda x: x[2])
print(f"least={pairs[:5]}")
print(f"most={pairs[-5:][::-1]}")
print("stable" if ciMean < 0.5 else f"unstable: wide CI, T=10 too short ({datetime.now().isoformat()})")
