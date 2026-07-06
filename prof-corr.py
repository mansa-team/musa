import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from scipy import stats
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

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=True, 
            fmt=".2f", annot_kws={"size": 7},
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Spearman ρ"})
plt.tight_layout()
plt.savefig("epsilon_correlation.png", dpi=150)
plt.close()


"""
to be used to compose the views of the black-litterman together with the xango investing scores

the main idea is to analyze the compostition from the profits of a stock over a 10 year period and compare stocks i,j to get a corr between their growths

x = year
y = profit

the profits would be scaled to be measured in a similar space
profits that grow together might indicate a market relation, in which the model would adapt the weight allocation based on that corr score

the evaluation of the difference from the growth of profits from t and t-1 or the direct comparasion between the y1 and y2, meaning that a MSE loss function would be used to evaluate a loss for the corr, the only problem is that, i fear that the scaler might not be able to adapt this properly or it would be able to catch the corr between stocks. or that a loss that is bigger between 2 points mean that the profit from the stock i might have been lower, not necessarily indicating a correlation

in a good makret, where stokcs grow their profits over and over, a profit correlation might be overfit, because the market will just be in a good phase, prioritizing stocks with less growth between each other, ruining the portfolios performance according to the mansas strategy


initial plan:

sum(mse) * -1 (inversion, lower loss better)
sigmoid or tahn * 0.5 (prob tahn because its derivative is higher than sigmoid's)
this would allow for a moldable corr model because of the k value in sigmoid and tahn to shift the threshold.

would compute the corr between all stocks i,j except when the ticker(i) == ticker(j)

so i can have

i - j
j - i

because the corr of them would be different, because mse is calculated with (y1 - y2)^2


scale the value to 100 base


possible problems:
- the corr might not show nothing because of the standardscaler, might just measure how big is the gap between one company and another in profits
- doesnt account for a healthy market


plan 2:
instead of measuring the profit values directly, measure the difference in growth between them in a t-1 and t time periods, convert it to % and use this as input data for y instead, comparing to the profit growth from another stock in the same period.

the rest stays the same, the only thing that changes is that

possible problems:
- doesnt account for a healthy market

possible solutions for the healthy market approach:
- consider, inside the picked of stocks through their score (> 60) the average growth in t-1 and t so it can evaluate a threshold for the mse. this would prob require a new loss function of to measure the diffrences between the percentages in growth from t-1 and t or some other equational approach



possible fixes to the algo:
mse measures the difference in distance between points y1 and y2, it doesnt measure the correlation in their movements, numpy have a builtin function for that, numpy.corrcoef that is able to measure the change in the profits overtime and should be used over the mse schema.

to evaluate the actual growth of a stock independent of its own linear trend and market noise, allowing a i, j comparasion between stocks, theres a need for the scipy.detrend linear function to be used to evaluate the the cyclical deviations in the i, j easily, this allows for co-cyclical evaluations that are still inherently associated with the market movement and to minimize the beta out of the data, analyzing both stocks real growth and trends between each other theres a need to remove the market association between the selected stocks we need to do regression in the detrended vector to extract the real identity of the function, for that:

the market factor is calculated to evaluate the commom signals shared between stocks using the sum of detrend from all stocks for that. the market factor is calcualted at every t point, representing the situation of the market at that time point M(t) that will be discounted further from each stock to make for a market noise free analysis
    M(t) = (1/N) sum{k=1}^{N} s_k(t)

after calculating the market factor for the stock, theres a need to find the beta (market corr factor), alpha (intercept) and epsilon (the stocks true relations, the idiosyncratic residual) for the stock, for that, regression is used to find the values for beta and alpha, so it finds the matching function for the detrend vector.
    s_i(t) = alpha_i + beta_i * M(t) + epsilon_i(t)

where the epsilon is the idiosyncratic residual, meaning its the stocks true moving identity and its associated signal. for the regression, the OLS function is used so the alpha and beta can be estimated.
to calculate the epsilon, we use
    epsilon_i(t) = s_i(t) - (alpha(t) + beta(t) * M(t))

after creating the vectors for the epsilon analysis, just numpy.corrcoef the stocks i and j, getting a final corr for the stocks profits.


possible idea: have this as an advanced user feature, letting the user compare 2 stocks and see if they are correlated by any way, also release this in the db?


patch 1:
instead of using one factor regression for the estimations, which can be overfitted easily and can be noisy somtimes because it only assumes that the market is the only driver for the brazilian market, the ideal proposal is to use some other measurement at the same time period to help me regression properly evaluate the market relevance's in the composal of the specific components from the regression. analyzing that, the addition of selic rates have been proposed, already being used in the workflow for the mansa project, would be a great help in the regression, reducing the noise from the correlations between epsilons in the stocks i, j and give a clearer overview on profits from sectors like banks that are heavily correlated with selic rates.
for that, we need to get a vector for selic, that is composed for the 10 years of data for selic, the same length as the market factor vector:
    S = S['valor'].resample('Y').mean()

for the regression, the new format is:
    s_i(t) = alpha_i + betaM_i * M(t) + betaS_i * S(t) + epsilon_i(t)

where betaS_i indicates how correlated the stock is to the change of the selic rates.

other important thing to notice is that, since the sample data is small, only 10 data points per stock to represent profits, we are working with a really constrained environment, for that, its proposed the usage of a different algorithm for regression other than OLS, because OLS with my 10 data points and k=2 only leaves only 7 degress of freedom for corretions and i need to treat N stocks, in which, OLS might suck to predict beta values, creating any relation between stocks, just noise.

the best alternative is to use a hierarchical bayesian regression algorithm that borrows information across all N stocks simultaneously. instead of fitting each stock independently (OLS), hierarchical bayesian learns a group-level distribution for beta across all stocks, then shrinks each stock's estimate toward the group mean. This works because all stocks contribute to estimating the shared distribution for beta, increasing the sample size from T=10 to T*N data points.

the current only built-in implementation for the model is inside the pymc lib, which is painfully slow because of monte carlo, as an alternative, the usage of statsmodels mixedlm that implements the same hierarchical structure via REML instead of MCMC sampling, that, at my data volume, is mathematically similar while running way faster (120s vs 10ms).


the final architecture also changes, instead of using the numpy.corrcoef, which is a pearson, theres a need to use the spearman correlation from scipy, that analyzes if a the vectors are actually moving in the same direction and is better at dealing with a low amount of data (10 data points), in which, for numpy.corrcoef, would inflate values according to some outlier or intense growth in profits at some point.

patch 2:
use np.log in the profits vector before using the detrend, to properly evaluate the exponential behavior of profits in a strategy, that, when using a detrend model, would remove the part in which the stock grows the most, tanking most scores.

used the market factor as the factor between all stocks instead of the subset, because the subset was too small, causing the linear regression to not converge


patch 3:

"""