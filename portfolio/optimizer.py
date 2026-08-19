import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from skfolio.pre_selection import DropCorrelated, DropZeroVariance, SelectKExtremes
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior
from skfolio.preprocessing import prices_to_returns
from skfolio.moments.covariance import LedoitWolf
from skfolio.measures import PerfMeasure

import time
start_time = time.time() 

np.random.seed(42)

selic = pd.DataFrame(requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json").json())

N_STOCKS = 18
RISK_FREE_RATE = float(selic['valor'].iloc[-1]) / 100

param_grid = {
    'risk_aversion': np.linspace(0.20, 0.60, 3),
    'l2_coef': np.logspace(-2, -1, 3),
}

df = pd.read_json("b3_stocks.json")
df = df[(df["VALUE INVESTING SCORE"] > 60) & (df["TICKER"].str.endswith("3"))]

price_data = [
    {"TICKER": row["TICKER"], "DATA": p["DATA"], "PRECO": p["PRECO"]}
    for _, row in df.iterrows() for p in row["COTACAO 10Y AJUSTADA"]
]

price_p = pd.DataFrame(price_data).pivot_table(index="DATA", columns="TICKER", values="PRECO", aggfunc="last")
price_p.index = pd.to_datetime(price_p.index, dayfirst=True)

X_prices_all = price_p.resample("ME").last().ffill().dropna(axis=1)
X_returns_candidates = prices_to_returns(X_prices_all)

all_tickers = X_returns_candidates.columns.tolist()

pre_selection = Pipeline([
    ("drop_zero_var", DropZeroVariance()),
    ("drop_corr", DropCorrelated(threshold=0.67, absolute=True)),
    ("select_k", SelectKExtremes(k=N_STOCKS, measure=PerfMeasure.MEAN, highest=True)),
])

tscv = TimeSeriesSplit(n_splits=3)
cv_results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_returns_candidates)):
    X_train_full, X_test_full = X_returns_candidates.iloc[train_idx], X_returns_candidates.iloc[test_idx]
    
    pre_selection.fit(X_train_full)
    
    steps = list(pre_selection.named_steps.values())
    fold_tickers = all_tickers
    for step in steps:
        fold_tickers = [t for t, m in zip(fold_tickers, step.get_support()) if m]
    
    X_train, X_test = X_train_full[fold_tickers], X_test_full[fold_tickers]
    scores = df.set_index("TICKER")["VALUE INVESTING SCORE"].reindex(fold_tickers)
    w_base = scores.values / scores.sum()
    
    for ra in param_grid['risk_aversion']:
        for l2 in param_grid['l2_coef']:
            try:
                model = MeanRisk(
                    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                    prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf()),
                    risk_aversion=ra,
                    l2_coef=l2,
                    target_weights=w_base,
                    max_weights=1.5/N_STOCKS,
                    min_weights=0.5/N_STOCKS,
                ).fit(X_train)
                
                test_returns = X_test @ model.weights_
                sharpe = (test_returns.mean() * 12 - RISK_FREE_RATE) / (test_returns.std() * np.sqrt(12))
                
                cv_results.append({'ra': ra, 'l2': l2, 'sharpe': sharpe})
            except:
                pass

cv_df = pd.DataFrame(cv_results)
best_ra, best_l2 = cv_df.groupby(['ra', 'l2'])['sharpe'].mean().idxmax()

print(f"ra: {best_ra:.2f} | l2: {best_l2:.4f} | sharpe: {cv_df[cv_df['ra']==best_ra]['sharpe'].mean():.4f}")

pre_selection.fit(X_returns_candidates)

steps = list(pre_selection.named_steps.values())
selected_tickers = all_tickers
for step in steps:
    selected_tickers = [t for t, m in zip(selected_tickers, step.get_support()) if m]

X_returns_selected = X_returns_candidates[selected_tickers]

scores = df.set_index("TICKER")["VALUE INVESTING SCORE"].reindex(selected_tickers)
w_base = scores.values / scores.sum()

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf()),
    risk_aversion=best_ra,
    l2_coef=best_l2,
    target_weights=w_base,
    max_weights=1.5/N_STOCKS,
    min_weights=0.5/N_STOCKS,
).fit(X_returns_selected)
weights = model.weights_

results = pd.DataFrame({
    "TICKER": X_returns_selected.columns,
    "WEIGHT": (weights / weights.sum() * 1000).round(2).astype(int)
}).sort_values("WEIGHT", ascending=False)

results["ALLOCATION (%)"] = (results["WEIGHT"] / results["WEIGHT"].sum() * 100).round(4)
print(results[["TICKER", "WEIGHT", "ALLOCATION (%)"]])

weights_series = pd.Series(weights, index=X_returns_selected.columns)
selected_weights = weights_series[selected_tickers].values

X_returns_selected = X_returns_selected[selected_tickers]

generate_graphs = True
if generate_graphs:
    corr_matrix = X_returns_selected.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, 
                fmt=".2f", xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Correlation"})
    plt.tight_layout()
    plt.savefig("corr_heatmap.png", dpi=150)
    plt.close()

    n_portfolios = 10000
    m = len(selected_tickers)
    random_weights = np.random.dirichlet(np.ones(m), n_portfolios)

    mu_returns = X_returns_selected.mean().values
    lw = LedoitWolf().fit(X_returns_selected)
    Sigma = lw.covariance_

    portfolio_returns = random_weights @ mu_returns
    portfolio_vols = np.array([np.sqrt(w @ Sigma @ w) for w in random_weights])

    opt_return = selected_weights @ mu_returns
    opt_vol = np.sqrt(selected_weights @ Sigma @ selected_weights)

    min_var_idx = np.argmin(portfolio_vols)
    min_var_vol = portfolio_vols[min_var_idx]
    min_var_return = portfolio_returns[min_var_idx]

    max_sharpe_idx = np.argmax((portfolio_returns - RISK_FREE_RATE) / portfolio_vols)
    max_sharpe_vol = portfolio_vols[max_sharpe_idx]
    max_sharpe_return = portfolio_returns[max_sharpe_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(portfolio_vols * np.sqrt(12) * 100, portfolio_returns * 12 * 100, 
               alpha=0.5, s=20, label="Random Portfolios", c="gray")
    ax.scatter(min_var_vol * np.sqrt(12) * 100, min_var_return * 12 * 100, 
               marker="*", s=200, c="green", label="Min Variance", zorder=5)
    ax.scatter(max_sharpe_vol * np.sqrt(12) * 100, max_sharpe_return * 12 * 100, 
               marker="*", s=200, c="blue", label="Max Sharpe", zorder=5)
    ax.scatter(opt_vol * np.sqrt(12) * 100, opt_return * 12 * 100, 
               marker="*", s=200, c="red", label="Chosen Portfolio", zorder=5)
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150)
    plt.close()

    print(f"\nreturn: {opt_return * 12 * 100:.2f}%")
    print(f"volatility: {opt_vol * np.sqrt(12) * 100:.2f}%")

print(f"\nelapsed: {time.time() - start_time:.2f}s")