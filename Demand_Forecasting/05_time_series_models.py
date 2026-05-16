#!/usr/bin/env python
# coding: utf-8

# # Time Series Models

# In[1]:


# !pip install pmdarima

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings('ignore')
random_state = 42


# In[2]:


df_train = pd.read_csv('train.csv', index_col='Date')
df_train.index = pd.to_datetime(df_train.index)

df_test = pd.read_csv('test.csv', index_col='Date')
df_test.index = pd.to_datetime(df_test.index)

target = 'Units Sold'

y_train = df_train[target]
X_train = df_train.drop(target, axis=1)

y_test = df_test[target]
X_test = df_test.drop(target, axis=1)

print(f"Train: {len(y_train)} rows | Test: {len(y_test)} rows")
print(f"Features: {X_train.shape[1]}")


# In[3]:


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator)

def weighted_mean_absolute_percentage_error(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    return numerator / denominator

def show_metrics(y_true, y_pred, label=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    sm   = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    wm   = weighted_mean_absolute_percentage_error(y_true, y_pred)
    print(f'{label}')
    print(f'  MAE:   {mae:.3f}')
    print(f'  RMSE:  {rmse:.3f}')
    print(f'  SMAPE: {sm:.3f}')
    print(f'  WMAPE: {wm:.3f}')
    plt.figure(figsize=(10, 3))
    plt.title(label or 'True vs Predicted')
    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='pred')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return {'label': label, 'MAE': mae, 'RMSE': rmse,
            'SMAPE': sm, 'WMAPE': wm}


# Expanding window helpers are used for one-step-ahead forecasting (expanding window protocol)

# In[4]:


def create_pred_list_arima(order, y_train, y_test):
    y_pred = pd.Series(index=y_test.index, dtype=float)
    for i in range(len(y_test)):
        temp_y_train = pd.concat([y_train, y_test[:i]])
        model = ARIMA(temp_y_train, order=order)
        model = model.fit()
        temp_y_pred = model.forecast(1)
        y_pred[temp_y_pred.index] = temp_y_pred
    return y_pred

def create_pred_list_arimax(order, X_train, y_train, X_test, y_test):
    y_pred = pd.Series(index=y_test.index, dtype=float)
    for i in range(len(y_test)):
        temp_X_train = pd.concat([X_train, X_test[:i]])
        temp_y_train = pd.concat([y_train, y_test[:i]])
        model = SARIMAX(
            temp_y_train,
            temp_X_train,
            order=order
        )
        model = model.fit(disp=False)
        temp_y_pred = model.forecast(1, exog=X_test[i:i+1])
        y_pred[temp_y_pred.index] = temp_y_pred
    return y_pred

def create_pred_list_ets(y_train, y_test):
    y_pred = pd.Series(index=y_test.index, dtype=float)
    for i in range(len(y_test)):
        temp_y_train = pd.concat([y_train, y_test[:i]])
        model = ETSModel(
            temp_y_train,
            error='add',
            trend='add',
            seasonal=None,
        )
        model = model.fit(disp=False)
        temp_y_pred = model.forecast(1)
        y_pred[temp_y_pred.index] = temp_y_pred
    return y_pred

def create_pred_list_exp_smooth(y_train, y_test):
    y_pred = pd.Series(index=y_test.index, dtype=float)
    for i in range(len(y_test)):
        temp_y_train = pd.concat([y_train, y_test[:i]])
        model = ExponentialSmoothing(
            temp_y_train,
            trend='add',
            seasonal=None,
        )
        model = model.fit()
        temp_y_pred = model.forecast(1)
        y_pred[temp_y_pred.index] = temp_y_pred
    return y_pred


# ## Auto Arima Settings

# In[5]:


# stationary=True is justified by full-panel ADF/KPSS analysis
# (88% of series stationary, S001/P0001 confirmed stationary)
# stepwise=True for speed

auto_arima_kwargs = dict(
    start_p=0,
    max_p=5,
    start_q=0,
    max_q=5,
    d=None,
    seasonal=False,
    stationary=True,
    stepwise=True,
    trace=False,
    n_fits=20,
    n_jobs=-1,
    scoring='mse',
)


# ## ARIMA

# In[6]:


# Auto-select order — no seasonal search
print("ARIMA")

auto_model = auto_arima(y_train, **auto_arima_kwargs)
print(f"Selected order: {auto_model.order}")

# One-step-ahead (expanding window)
print("\nOne-step-ahead:")
y_pred_arima_exp = create_pred_list_arima(
    auto_model.order, y_train, y_test)
r_arima_exp = show_metrics(y_test, y_pred_arima_exp,
                            f'ARIMA{auto_model.order} — expanding')

# Batch prediction
print("\nBatch:")
model = ARIMA(y_train, order=auto_model.order)
model = model.fit()
y_pred_arima_batch = model.forecast(len(y_test))
r_arima_batch = show_metrics(y_test, y_pred_arima_batch,
                              f'ARIMA{auto_model.order} — batch')


# ## ARIMAX

# In[7]:


auto_model_x = auto_arima(y_train, X_train, **auto_arima_kwargs)
print(f"Selected order: {auto_model_x.order}")

print("\nBatch:")
model_x = SARIMAX(y_train, X_train, order=auto_model_x.order)
model_x = model_x.fit(disp=False)
y_pred_arimax_batch = model_x.forecast(len(y_test), exog=X_test)
y_pred_arimax_batch = pd.Series(
    np.clip(y_pred_arimax_batch, 0, None),
    index=y_test.index
)
r_arimax_batch = show_metrics(y_test, y_pred_arimax_batch,
                               f'ARIMAX{auto_model_x.order} — batch')


# ## ETS

# In[8]:


# Seasonal variants excluded: panel analysis shows no significant 
# seasonality in S001/P0001 or majority of panel series
print("ETS — non-seasonal")

# One-step-ahead
print("\nOne-step-ahead:")
y_pred_ets_exp = create_pred_list_ets(y_train, y_test)
r_ets_exp = show_metrics(y_test, y_pred_ets_exp,
                          'ETS (non-seasonal) — expanding')

# Batch
print("\nBatch:")
model_ets = ETSModel(
    y_train,
    error='add',
    trend='add',
    seasonal=None,
)
model_ets = model_ets.fit(disp=False)
y_pred_ets_batch = model_ets.forecast(len(y_test))
r_ets_batch = show_metrics(y_test, y_pred_ets_batch,
                            'ETS (non-seasonal) — batch')


# ## HOLT-WINTERS

# In[9]:


# Seasonal variants excluded: same justification as ETS above
print("Holt-Winters — non-seasonal")

# One-step-ahead
print("\nOne-step-ahead:")
y_pred_hw_exp = create_pred_list_exp_smooth(y_train, y_test)
r_hw_exp = show_metrics(y_test, y_pred_hw_exp,
                         'Holt-Winters (non-seasonal) — expanding')

# Batch
print("\nBatch:")
model_hw = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal=None,
)
model_hw = model_hw.fit()
y_pred_hw_batch = model_hw.forecast(len(y_test))
r_hw_batch = show_metrics(y_test, y_pred_hw_batch,
                           'Holt-Winters (non-seasonal) — batch')


# In[10]:


results = [
    r_arima_exp,    r_arima_batch,
    r_arimax_batch,               
    r_ets_exp,      r_ets_batch,
    r_hw_exp,       r_hw_batch,
]

summary_df = pd.DataFrame(results)
print(summary_df.round(3).to_string(index=False))
summary_df.to_csv('classical_model_results.csv', index=False)

