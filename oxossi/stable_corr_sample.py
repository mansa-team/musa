import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.signal import detrend
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
baseDir = os.path.dirname(os.path.abspath(__file__))
apiBase = "http://localhost:3200"
nowYear = datetime.now().year
yearsRange = [str(y) for y in range(nowYear - 10, nowYear)]
fundField = "XANGO INVESTING SCORE"
histFields = "LUCRO LIQUIDO,RECEITA LIQUIDA,MARGEM LIQUIDA"
chanList = [c.strip() for c in histFields.split(",")]
rng = np.random.default_rng(42)
selicRaw = pd.DataFrame(requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json").json())
selicRaw["valor"] = selicRaw["valor"].astype(float)
selicRaw["data"] = pd.to_datetime(selicRaw["data"], dayfirst=True)
selicFrame = selicRaw.set_index("data").resample("YE").mean()
selicFrame.index = selicFrame.index.year.astype(str)
selicFrame = selicFrame.reindex(yearsRange)
selicVals = selicFrame["valor"].values.astype(float)
fundFrame = pd.DataFrame(requests.get(f"{apiBase}/stocks/fundamental", params={"fields": fundField, "dates": "2026-06-29"}).json()["data"])
candFrame = fundFrame[(fundFrame[fundField] > 60) & (fundFrame["TICKER"].str.endswith("3"))].sort_values(fundField, ascending=False)
allTickers = ",".join(fundFrame[fundFrame["TICKER"].str.endswith("3")]["TICKER"].tolist())
histFrame = pd.DataFrame(requests.get(f"{apiBase}/stocks/historical", params={"fields": histFields, "search": allTickers}).json()["data"]).drop(columns=["NOME"], errors="ignore").set_index("TICKER")
def chanMat(frame, chan):
    cols = [c for c in frame.columns if c.startswith(chan)]
    yearOf = {c: "".join(ch for ch in c if ch.isdigit())[-4:] for c in cols}
    keep = {c: y for c, y in yearOf.items() if y in yearsRange}
    sub = frame[list(keep)].rename(columns=keep)[yearsRange].apply(pd.to_numeric, errors="coerce").dropna()
    return sub
chanMats = {c: chanMat(histFrame, c) for c in chanList}
commonSet = set(candFrame["TICKER"]) & set.intersection(*[set(m.index) for m in chanMats.values()])
llMat = chanMats["LUCRO LIQUIDO"].loc[sorted(commonSet)]
llCorr = pd.DataFrame(np.log1p(np.maximum(llMat.values.astype(float), 0)), index=llMat.index).T.corr(method="spearman")
scoreMap = dict(zip(candFrame["TICKER"], candFrame[fundField]))
ranked = sorted(llMat.index, key=lambda t: -scoreMap.get(t, 0))
keptTickers, droppedTickers = [], []
for tick in ranked:
    if any(abs(llCorr.loc[tick, k]) > 0.7 for k in keptTickers):
        droppedTickers.append(tick)
    else:
        keptTickers.append(tick)
keptTickers = sorted(keptTickers[:12], key=lambda t: -scoreMap.get(t, 0)) if len(keptTickers) > 12 else keptTickers
print(f"kept({len(keptTickers)}): {keptTickers}")
print(f"dropped({len(droppedTickers)}): {droppedTickers}")
nStocks, nYears = len(keptTickers), len(yearsRange)
chanVals = [chanMats[c].loc[keptTickers].values.astype(float) for c in chanList]
def pooledEps(logVals):
    detr = np.apply_along_axis(detrend, 1, logVals)
    mFac = detr.mean(axis=0)
    sStd = (selicVals - selicVals.mean()) / selicVals.std() * detr.std()
    regDf = pd.DataFrame({"y": detr.flatten(), "M": np.tile(mFac, nStocks), "S": np.tile(sStd.flatten(), nStocks)})
    fitRes = smf.ols("y ~ M + S", data=regDf).fit()
    eps = detr - (fitRes.params["Intercept"] + fitRes.params["M"] * mFac + fitRes.params["S"] * sStd.flatten())
    return eps, detr, mFac, sStd.flatten()
epsList, detrList, mList, sList = [], [], [], []
for vals in chanVals:
    eVals, dVals, mVals, sVals = pooledEps(np.log1p(np.maximum(vals, 0)))
    epsList.append(eVals)
    detrList.append(dVals)
    mList.append(mVals)
    sList.append(sVals)
epsStack = np.hstack(epsList)
rawCorr, _ = spearmanr(epsStack.T)
rawCorr = np.atleast_2d(np.nan_to_num(rawCorr, nan=0.0))
np.fill_diagonal(rawCorr, 1.0)
triMask = np.triu_indices(nStocks, k=1)
constTarget = rawCorr[triMask].mean()
offMat = np.full_like(rawCorr, constTarget)
np.fill_diagonal(offMat, 1.0)
dVal = 0.5
shrunkCorr = (1 - dVal) * rawCorr + dVal * offMat
print(f"raw mean={rawCorr[triMask].mean():.3f} shrunk mean={shrunkCorr[triMask].mean():.3f} const={constTarget:.3f}")
betaVec = np.array([np.polyfit(mList[0], detrList[0][i], 1)[0] for i in range(nStocks)])
scoreVec = np.array([scoreMap[t] for t in keptTickers], dtype=float)
betaGap = np.abs(betaVec[:, None] - betaVec[None, :]) / (np.ptp(betaVec) + 1e-12)
scoreGap = np.abs(scoreVec[:, None] - scoreVec[None, :]) / (np.ptp(scoreVec) + 1e-12)
sameGroup = np.array([[1.0 if keptTickers[i][0] == keptTickers[j][0] else 0.0 for j in range(nStocks)] for i in range(nStocks)])
proxyCorr = np.clip(0.5 - 0.3 * betaGap - 0.2 * scoreGap + 0.2 * sameGroup, -1, 1)
np.fill_diagonal(proxyCorr, 1.0)
finalCorr = (shrunkCorr + proxyCorr) / 2
nBoot = 500
bootMats = np.zeros((nBoot, nStocks, nStocks))
for bIdx in range(nBoot):
    yearIdx = rng.choice(nYears * len(chanList), nYears * len(chanList), replace=True)
    bootStack = epsStack[:, yearIdx]
    cb, _ = spearmanr(bootStack.T)
    cb = np.atleast_2d(np.nan_to_num(cb, nan=0.0))
    np.fill_diagonal(cb, 1.0)
    shrunkB = (1 - dVal) * cb + dVal * offMat
    bootMats[bIdx] = (shrunkB + proxyCorr) / 2
bootZ = np.arctanh(np.clip(bootMats, -0.999, 0.999))
ciLo = np.tanh(np.percentile(bootZ, 2.5, axis=0))
ciHi = np.tanh(np.percentile(bootZ, 97.5, axis=0))
ciWidth = ciHi - ciLo
np.fill_diagonal(ciWidth, 0)
stableMask = (ciLo > 0) | (ciHi < 0)
sparseCorr = np.where(stableMask, finalCorr, 0.0)
np.fill_diagonal(sparseCorr, 1.0)
nPairs = len(triMask[0])
nKept = int(stableMask[triMask].sum())
print(f"stable pairs kept={nKept}/{nPairs} meanCI={ciWidth[triMask].mean():.3f}")
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
for ax, mat, ttl, cmap, ctr in [(axes[0], pd.DataFrame(finalCorr, index=keptTickers, columns=keptTickers), "Final dense (shrunk+proxy)/2", "coolwarm", 0), (axes[1], pd.DataFrame(sparseCorr, index=keptTickers, columns=keptTickers), f"Sparse stable ({nKept}/{nPairs})", "coolwarm", 0), (axes[2], pd.DataFrame(ciWidth, index=keptTickers, columns=keptTickers), "CI width (final, time-boot)", "YlOrRd", None)]:
    sns.heatmap(mat, cmap=cmap, center=ctr, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title(ttl)
plt.tight_layout()
plt.savefig(os.path.join(baseDir, "stable_corr.png"), dpi=150)
plt.close()
pairStats = [(keptTickers[i], keptTickers[j], finalCorr[i, j], ciWidth[i, j]) for i, j in zip(*triMask)]
pairStats.sort(key=lambda x: x[3])
print(f"top5 stable={pairStats[:5]}")
print(f"least5 stable={pairStats[-5:][::-1]}")
print("stable" if ciWidth[triMask].mean() < 0.5 and nKept / max(nPairs, 1) > 0.3 else "unstable")
