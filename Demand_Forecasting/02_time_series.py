#!/usr/bin/env python
# coding: utf-8

# # FORMATION OF TIME SERIES

# In[15]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats.mstats import kendalltau
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


# In[16]:


warnings.filterwarnings('ignore')


# In[17]:


dataset_path = 'data_enriched.csv'
df = pd.read_csv(dataset_path)


# ## Preparing the temporary structure

# Checking the regularity of the time grid

# In[7]:


df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Store ID", "Product ID"], ignore_index=True)

dates = df["Date"].unique()
if len(dates) > 1:
    steps = pd.Series(dates).diff().dropna().dt.days
    unique_steps = steps.unique()
    if len(unique_steps) == 1:
        print(f"The time grid is regular: step = {int(unique_steps[0])} d.")
    else:
        print(f"The time grid is irregular; steps (days): {unique_steps}")


# Formation of panel time series: (Store ID, Product ID) → Units Sold time series

# In[8]:


df_panel = (
    df.set_index(["Store ID", "Product ID", "Date"])
    .sort_index()["Units Sold"]
)


# ## Time series analysis (EDA)

# In[9]:


def show_series_analysis(series, title='', alpha=0.05, lags=30):
    # Plots
    decomposition = seasonal_decompose(series, period=lags)
    
    fig, axes = plt.subplots(6, 1, figsize=(30, 20))
    plt.suptitle(title, fontsize=16, y=1)
    
    decomposition.observed.plot(ax=axes[0], title='Original series')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
    decomposition.resid.plot(ax=axes[3], title='Residuals')
    plot_acf(series, lags=lags, title='Autocorrelation', ax=axes[4])
    plot_pacf(series, lags=lags, title='Partial autocorrelation', ax=axes[5])
    plt.tight_layout()
    plt.show()
    
    tau, p = kendalltau(range(len(series)), series)
    if p < alpha:
        print("The Mann-Kendall test for trend is statistically significant.")
        print(f"τ = {tau:.3f} (the closer to 1/-1, the stronger the trend)")
    else:
        print("The Mann-Kendall test for trend is not statistically significant.")


    # Tests
    adf_pvalue = adfuller(series, autolag='AIC')[1]
    kpss_pvalue = kpss(series)[1]

    if adf_pvalue < alpha and kpss_pvalue >= alpha:
        print('ADF and KPSS tests showed stationarity')
    elif adf_pvalue < alpha and kpss_pvalue >= alpha:
        print('ADF and KPSS tests showed non-stationarity')
    else:
        print('ADF and KPSS tests failed to detect stationarity')

    
    critical_value = 1.96 / np.sqrt(len(series))

    acf_values = acf(series, nlags=lags, fft=False)
    pacf_values = pacf(series, nlags=lags)

    
    print('Significant ACF lags:')
    for lag in range(1, min(lags, len(acf_values))):
        value = acf_values[lag]
        if abs(value) > critical_value:
            print(f'{lag}: {value:0.3f}')
        
    print('Significant PACF lags:')
    for lag in range(1, min(lags, len(pacf_values))):
        value = pacf_values[lag]
        if abs(value) > critical_value:
            print(f'{lag}: {value:0.3f}')


# I will determine the presence of a trend using the Mann-Kendall test.
# 
# Stationarity will be determined using the ADF and KPSS tests.
# 
# Seasonality will be determined using the ACF and PACF tests.

# In[10]:


def show_series_analysis(series, title="", alpha=0.05, lags=30, period=None):
    series = series.dropna()
    n = len(series)
    if n < 2:
        print("The series is too short for analysis.")
        return

    lags = min(lags, n - 1)
    if period is None:
        period = min(30, n // 2)
    if period < 3:
        period = 3
    if period % 2 == 0:
        period -= 1

    try:
        decomposition = seasonal_decompose(series, period=period)
    except Exception as e:
        print(f"Decomposition is missed: {e}")
        decomposition = None

    fig, axes = plt.subplots(6, 1, figsize=(12, 10))
    fig.suptitle(title or "Time series", fontsize=14, y=1.02)

    if decomposition is not None:
        decomposition.observed.plot(ax=axes[0], title="Original series")
        decomposition.trend.plot(ax=axes[1], title="Trend")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
        decomposition.resid.plot(ax=axes[3], title="Residuals")
    else:
        axes[0].plot(series)
        axes[0].set_title("Original series")
        for ax in axes[1:4]:
            ax.set_visible(False)

    plot_acf(series, lags=lags, ax=axes[4], title="ACF")
    plot_pacf(series, lags=lags, ax=axes[5], title="PACF")
    plt.tight_layout()
    plt.show()

    # Kendall tau (trend)
    tau, p = kendalltau(np.arange(n), series.values)
    if p < alpha:
        print("Mann-Kendall test: trend is statistically significant")
        print(f"  τ = {tau:.3f} (closer to ±1 - stronger trend)")
    else:
        print("Mann-Kendall test: trend is not statistically significant")

    adf_pvalue = adfuller(series, autolag="AIC")[1]
    kpss_pvalue = kpss(series, regression="c")[1]
    if adf_pvalue < alpha and kpss_pvalue >= alpha:
        print("ADF and KPSS: stationary series")
    elif adf_pvalue >= alpha and kpss_pvalue < alpha:
        print("ADF and KPSS: non-stationary series")
    else:
        print("ADF and KPSS: the result is contradictory or ambiguous")
    print(f"  ADF p-value: {adf_pvalue:.4f}, KPSS p-value: {kpss_pvalue:.4f}")

    critical = 1.96 / np.sqrt(n)
    acf_vals = acf(series, nlags=lags, fft=False)
    pacf_vals = pacf(series, nlags=lags)
    sig_acf = [lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) > critical]
    sig_pacf = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > critical]
    print("Significant ACF lags:", sig_acf if sig_acf else "none")
    print("Significant PACF lags:", sig_pacf if sig_pacf else "none")


# In[11]:


SERIES_TO_ANALYZE = [
    ("S001", "P0001"),  # conclusions: no trend, stationarity exists, no seasonality
    ("S005", "P0020"),  # Conclusions: no trend, stationarity exists, seasonality is 4 and 27 days
    ("S001", "P0002"),  # conclusions: weak downward trend, non-stationarity, seasonality of 1 and ~11 days
]


if __name__ == "__main__":
    for store_id, product_id in SERIES_TO_ANALYZE:
        title = f"Store ID={store_id}, Product ID={product_id}"
        show_series_analysis(df_panel[store_id, product_id], title=title)
        print()


# ### S001, P0001

# In[12]:


store_id = 'S001'
product_id = 'P0001'
title = f'Plot: Store ID={store_id}, Product ID={product_id}'

show_series_analysis(df_panel[store_id, product_id], title)


# ### S005, P0020

# In[13]:


store_id = 'S005'
product_id = 'P0020'
title = f'Plot: Store ID={store_id}, Product ID={product_id}'

show_series_analysis(df_panel[store_id, product_id], title)


# ### S001, P0002

# In[13]:


store_id = 'S001'
product_id = 'P0002'
title = f'Plot: Store ID={store_id}, Product ID={product_id}'

show_series_analysis(df_panel[store_id, product_id], title)


# ## Multi-Series Statistical Analysis (100 series)

# In[14]:


from statsmodels.stats.diagnostic import acorr_ljungbox

df_enriched = pd.read_csv('data_enriched.csv')
df_enriched['Date'] = pd.to_datetime(df_enriched['Date'])
df_enriched = df_enriched.sort_values(
    ['Store ID', 'Product ID', 'Date']
).reset_index(drop=True)

def analyze_series(series, store, product, category, region):
    s = series.values
    n = len(s)
    res = {'store': store, 'product': product,
           'category': category, 'region': region, 'n': n,
           'mean': np.mean(s), 'std': np.std(s),
           'cv': np.std(s) / np.mean(s) if np.mean(s) > 0 else np.nan}

    # Mann-Kendall
    s_mk = sum(
        np.sign(s[j] - s[i])
        for i in range(n - 1) for j in range(i + 1, n)
    )
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z_mk = (s_mk - np.sign(s_mk)) / np.sqrt(var_s) if s_mk != 0 else 0
    res['mk_trend']   = abs(z_mk) > 1.96
    res['trend_dir']  = 'up' if z_mk > 0 else 'down' if z_mk < 0 else 'none'
    res['kendall_tau'] = s_mk / (n * (n - 1) / 2)

    # ADF + KPSS
    try:
        adf_p = adfuller(s, autolag='AIC')[1]
        kpss_p = kpss(s, regression='c', nlags='auto')[1]
        res['adf_p'] = adf_p
        res['kpss_p'] = kpss_p
        if adf_p < 0.05 and kpss_p >= 0.05:
            res['stationarity'] = 'stationary'
        elif adf_p >= 0.05 and kpss_p < 0.05:
            res['stationarity'] = 'non_stationary'
        else:
            res['stationarity'] = 'inconclusive'
    except:
        res['adf_p'] = res['kpss_p'] = np.nan
        res['stationarity'] = 'unknown'

    # ACF significant lags
    try:
        acf_vals = acf(s, nlags=30, fft=True)
        ci = 1.96 / np.sqrt(n)
        sig = [i for i in range(1, 31) if abs(acf_vals[i]) > ci]
        res['n_sig_acf']       = len(sig)
        res['weekly_seasonal'] = 7 in sig
        res['monthly_seasonal'] = any(l >= 28 for l in sig)
    except:
        res['n_sig_acf'] = 0
        res['weekly_seasonal'] = res['monthly_seasonal'] = False

    # Ljung-Box
    try:
        lb_p = acorr_ljungbox(s, lags=[10], return_df=True)['lb_pvalue'].values[0]
        res['ljungbox_p']  = lb_p
        res['arima_useful'] = lb_p < 0.05
    except:
        res['ljungbox_p']  = np.nan
        res['arima_useful'] = False

    return res

# Run on all 100 series
results = []
for (store, product), grp in df_enriched.groupby(['Store ID', 'Product ID']):
    r = analyze_series(
        grp.sort_values('Date')['Units Sold'],
        store, product,
        grp['Category'].iloc[0], grp['Region'].iloc[0]
    )
    results.append(r)

analysis_df = pd.DataFrame(results)

# Assign profiles
analysis_df['profile'] = 'other'
analysis_df.loc[
    (analysis_df['stationarity'] == 'stationary') &
    (~analysis_df['mk_trend']) & (~analysis_df['arima_useful']),
    'profile'] = 'stationary_no_autocorr'
analysis_df.loc[
    (analysis_df['stationarity'] == 'stationary') &
    (analysis_df['arima_useful']),
    'profile'] = 'stationary_autocorr'
analysis_df.loc[analysis_df['mk_trend'], 'profile'] = 'trending'
analysis_df.loc[
    analysis_df['stationarity'] == 'inconclusive',
    'profile'] = 'inconclusive'

print("\nProfile distribution:")
print(analysis_df['profile'].value_counts())
print(f"\nSeries where ARIMA is meaningful: {analysis_df['arima_useful'].sum()}")
print(f"Series with weekly seasonality:   {analysis_df['weekly_seasonal'].sum()}")

# Stratified sample
selected = []
for profile, n_pick in [
    ('stationary_no_autocorr', 3),
    ('stationary_autocorr', 4),
    ('trending', 3),
    ('inconclusive', 2),
]:
    pool = analysis_df[analysis_df['profile'] == profile]
    k = min(n_pick, len(pool))
    if k > 0:
        selected.append(pool.sample(k, random_state=42))
        print(f"  {profile}: selected {k} of {len(pool)}")

selected_df = pd.concat(selected, ignore_index=True)
print(f"\nTotal selected: {len(selected_df)} series")
print(selected_df[['store', 'product', 'profile',
                    'stationarity', 'arima_useful',
                    'n_sig_acf', 'mean']].to_string(index=False))

analysis_df.to_csv('series_analysis.csv', index=False)
selected_df.to_csv('selected_series.csv', index=False)

