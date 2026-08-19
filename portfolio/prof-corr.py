import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from scipy.signal import detrend
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

import requests

current_year = datetime.now().year
years_range = [str(y) for y in range(current_year - 10, current_year)]

# selic data
selic = pd.DataFrame(requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json").json())
selic['valor'] = selic['valor'].astype(float)
selic['data'] = pd.to_datetime(selic['data'])

selic = selic.set_index('data')
selic = selic.resample('YE').mean()
selic.index = selic.index.year

selic.index = selic.index.astype(str)
selic = selic.reindex(years_range)

# ticker data
tickers = pd.DataFrame(requests.get('http://localhost:3200/stocks/fundamental?fields=XANGO INVESTING SCORE&dates=2026-06-29').json()['data'])

tickers = tickers[(tickers["XANGO INVESTING SCORE"] > 0) & (tickers["TICKER"].str.endswith("3"))]
selected_tickers_df = tickers[(tickers["XANGO INVESTING SCORE"] > 60) & (tickers["TICKER"].str.endswith("3"))]
tickers = ", ".join(tickers['TICKER'].to_list())

profits = pd.DataFrame(requests.get(f'http://localhost:3200/stocks/historical?fields=LUCRO LIQUIDO&search={tickers}').json()['data'])
profits = profits.drop(columns={"NOME"}).set_index('TICKER')
profits.columns = profits.columns.str.extract(r'(\d+)')[0]

profits_10y = profits[years_range].dropna()

selected_tickers = [t for t in selected_tickers_df['TICKER'].tolist() if t in profits_10y.index]

# data engineering
log_profits = np.log(np.maximum(profits_10y.values.astype(float), 1))
detrended = np.apply_along_axis(detrend, 1, log_profits)

selected_log_profits = np.log(profits_10y.loc[selected_tickers].values.astype(float) + 1)
selected_detrended = np.apply_along_axis(detrend, 1, selected_log_profits)

selic_standardized = (selic.values - selic.values.mean()) / selic.values.std()
selic_standardized = selic_standardized * detrended.std()

M = detrended.mean(axis=0)

# regression
N_stocks = selected_detrended.shape[0]
T_years = selected_detrended.shape[1]

regression_df = pd.DataFrame({
    'detrended_value': selected_detrended.flatten(),
    'M': np.tile(M, N_stocks),
    'S': np.tile(selic_standardized.flatten(), N_stocks),
    'tickers': np.repeat(selected_tickers, T_years),
    'time_idx': np.tile(np.arange(T_years), N_stocks)
})

result = smf.ols(
    'detrended_value ~ M + S',
    data=regression_df
).fit()

# epsilon extraction
alpha = result.params['Intercept']
beta_m_fixed = result.params['M']
beta_s_fixed = result.params['S']

epsilon = np.zeros_like(selected_detrended)
for i, ticker in enumerate(selected_tickers):
    epsilon[i] = selected_detrended[i] - (alpha + beta_m_fixed * M + beta_s_fixed * selic_standardized.flatten())

betas_m = np.full(N_stocks, beta_m_fixed) 

# spearman correlation
corr, pval = spearmanr(epsilon.T)
np.fill_diagonal(corr, 1.0)

corr_df = pd.DataFrame(corr, index=selected_tickers, columns=selected_tickers)

# spearman + shrinkage toward identity
SHRINK = 0.3
corr_shrunk = (1 - SHRINK) * corr + SHRINK * np.eye(N_stocks)
corr_shrunk_df = pd.DataFrame(corr_shrunk, index=selected_tickers, columns=selected_tickers)

# bootstrap CI
N_BOOT = 500
rng = np.random.default_rng(42)
boot_matrices = np.zeros((N_BOOT, N_stocks, N_stocks))
for b in range(N_BOOT):
    idx = rng.choice(N_stocks, N_stocks, replace=True)
    eb = epsilon[idx]
    cb, _ = spearmanr(eb.T)
    boot_matrices[b] = cb

# Fisher z-transform for CIs
boot_z = np.arctanh(np.clip(boot_matrices, -0.999, 0.999))
ci_lo = np.tanh(np.percentile(boot_z, 2.5, axis=0))
ci_hi = np.tanh(np.percentile(boot_z, 97.5, axis=0))
ci_width = ci_hi - ci_lo
np.fill_diagonal(ci_width, 0)

ci_width_df = pd.DataFrame(ci_width, index=selected_tickers, columns=selected_tickers)

# heatmap
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 7},
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Spearman rho"}, ax=axes[0])
axes[0].set_title("Raw Spearman (epsilon)")

sns.heatmap(corr_shrunk_df, cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 7},
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Spearman rho"}, ax=axes[1])
axes[1].set_title(f"Shrunk (alpha={SHRINK})")

sns.heatmap(ci_width_df, cmap="YlOrRd", annot=True,
            fmt=".2f", annot_kws={"size": 7},
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "95% CI Width"}, ax=axes[2])
axes[2].set_title("Bootstrap 95% CI Width")
plt.tight_layout()
plt.savefig("epsilon_correlation_with_ci.png", dpi=150)
plt.close()

# summary stats
upper_raw = corr[np.triu_indices(N_stocks, k=1)]
upper_shrunk = corr_shrunk[np.triu_indices(N_stocks, k=1)]
upper_ci = ci_width[np.triu_indices(N_stocks, k=1)]
print(f"Raw Spearman   — mean: {np.mean(upper_raw):.3f}, median: {np.median(upper_raw):.3f}")
print(f"Shrunk (a={SHRINK}) — mean: {np.mean(upper_shrunk):.3f}, median: {np.median(upper_shrunk):.3f}")
print(f"CI width       — mean: {np.mean(upper_ci):.3f}, median: {np.median(upper_ci):.3f}, max: {np.max(upper_ci):.3f}")