#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats.mstats import kendalltau
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


dataset_path = 'final_version_A.csv'
df = pd.read_csv(dataset_path)
df


# # ГЛАВА 2. ФОРМИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ

# ## Подготовка временной структуры

# ### Сортировка данных по Date

# In[4]:


df['Date'] = pd.to_datetime(df['Date'])


# In[5]:

df = df.sort_values(['Date', 'Store ID', 'Product ID'], ignore_index=True)
df 


# ### Проверка регулярности временной сетки

# In[6]:


first_d = df['Date'].unique()[0]
diffs = set()

for d in df['Date'].unique()[1:]:
    diffs.add(d-first_d)
    first_d = d


# In[7]:


diffs


# Временная сетка регулярна: шаг равен 1 дню

# In[8]:

# # Изменения функции агрегации для столбцов
# daily_agg = df.groupby('Date').agg({
#     'Units Sold': 'sum',
#     'Units Ordered': 'sum',
#     'Demand Forecast': 'sum',
#     'Inventory Level': 'mean',
#     'Price': 'mean',
#     'Discount': 'mean',
#     'Weather Condition': lambda x: x.mode()[0] if len(x.mode()) > 0 else None, # выбирается самое частое
#     'Holiday/Promotion': lambda x: 1 if any(x == 1) else 0,  # стояла ли хотя бы одна 1 в столбце
#     'Competitor Pricing': 'mean',
#     'Seasonality': lambda x: x.mode()[0] if len(x.mode()) > 0 else None, # выбирается самое частое
# }).reset_index()

# daily_agg


# ### Формирование panel time series: (Store ID, Product ID) → временной ряд Units Sold

# In[9]:

df_panel = df.set_index(['Store ID', 'Product ID', 'Date']).sort_index()['Units Sold']
df_panel


# ## Анализ временных рядов (EDA)

# In[10]:


def show_series_analysis(series, title='', alpha=0.05, lags=30):
    # Графики
    decomposition = seasonal_decompose(series, period=lags)
    
    fig, axes = plt.subplots(6, 1, figsize=(30, 20))
    plt.suptitle(title, fontsize=16, y=1)
    
    decomposition.observed.plot(ax=axes[0], title='Исходный ряд')
    decomposition.trend.plot(ax=axes[1], title='Тренд')
    decomposition.seasonal.plot(ax=axes[2], title='Сезонность')
    decomposition.resid.plot(ax=axes[3], title='Остатки')
    plot_acf(series, lags=lags, title='Автокорреляция', ax=axes[4])
    plot_pacf(series, lags=lags, title='Частичная автокорреляция', ax=axes[5])
    plt.tight_layout()
    plt.show()
    
    tau, p = kendalltau(range(len(series)), series)
    if p < alpha:
        print("Тест Манна-Кендалла на наличие тренда статистически значим")
        print(f"τ = {tau:.3f} (чем ближе к 1/-1, тем сильнее тренд)")
    else:
        print("Тест Манна-Кендалла на наличие тренда статистически не значим")


    # Тесты
    adf_pvalue = adfuller(series, autolag='AIC')[1]
    kpss_pvalue = kpss(series)[1]

    if adf_pvalue < alpha and kpss_pvalue >= alpha:
        print('Тесты ADF и KPSS показали стационарность')
    elif adf_pvalue < alpha and kpss_pvalue >= alpha:
        print('Тесты ADF и KPSS показали нестационарность')
    else:
        print('Тесты ADF и KPSS не смогли определить стационарность')

    
    critical_value = 1.96 / np.sqrt(len(series))

    acf_values = acf(series, nlags=lags, fft=False)
    pacf_values = pacf(series, nlags=lags)

    
    print('Значимые лаги ACF:')
    for lag in range(1, min(lags, len(acf_values))):
        value = acf_values[lag]
        if abs(value) > critical_value:
            print(f'{lag}: {value:0.3f}')
        
    print('Значимые лаги PACF:')
    for lag in range(1, min(lags, len(pacf_values))):
        value = pacf_values[lag]
        if abs(value) > critical_value:
            print(f'{lag}: {value:0.3f}')


# Наличие тренда будем определять по критерию Манна-Кендалла. 
# 
# Стационарность будем определять по тестам ADF и KPSS. 
# 
# Сезонность будем определять по ACF и PACF.

# ### S001, P0001

# In[11]:


store_id = 'S001'
product_id = 'P0001'
title = f'График: Store ID={store_id}, Product ID={product_id}'

show_series_analysis(df_panel[store_id, product_id], title)


# Выводы: 
# 
# * тренд: нет
# * стационарность: есть
# * сезонность: нет

# ### S005, P0020

# In[12]:


store_id = 'S005'
product_id = 'P0020'
title = f'График: Store ID={store_id}, Product ID={product_id}'

show_series_analysis(df_panel[store_id, product_id], title)


# Выводы: 
# 
# * тренд: нет
# * стационарность: есть
# * сезонность: 4 и 27 дней

# ### S001, P0002

# In[13]:


store_id = 'S001'
product_id = 'P0002'
title = f'График: Store ID={store_id}, Product ID={product_id}'

show_series_analysis(df_panel[store_id, product_id], title)


# Выводы: 
# 
# * тренд: очень слабый нисходящий
# * стационарность: нет
# * сезонность: 1 день и возможно 11 дней
