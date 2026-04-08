#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pmdarima


# In[2]:


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


# In[3]:


random_state = 42


# In[4]:


warnings.filterwarnings('ignore')


# In[5]:


df_train = pd.read_csv('train.csv', index_col='Date')
df_train.index = pd.to_datetime(df_train.index)
df_train


# In[6]:


df_test = pd.read_csv('test.csv', index_col='Date')
df_test.index = pd.to_datetime(df_test.index)
df_test


# In[7]:


target = 'Units Sold'

y_train = df_train[target]
X_train = df_train.drop(target, axis=1)

y_test = df_test[target]
X_test = df_test.drop(target, axis=1)


# # ГЛАВА 5. МОДЕЛИ ВРЕМЕННЫХ РЯДОВ

# In[8]:


def create_pred_list_arima(order, y_train, y_test, seasonal_order=(0, 0, 0, 0)):
    y_pred = pd.Series(index=y_test.index)

    for i in range(len(y_test)):
        temp_y_train = pd.concat([y_train, y_test[:i]])
        temp_y_test = y_test[i:]
    
        model = ARIMA(
            temp_y_train, 
            order=order, 
            seasonal_order=seasonal_order
        )
        model = model.fit()
        temp_y_pred = model.forecast(1)
        
        y_pred[temp_y_pred.index] = temp_y_pred

    return y_pred


# In[9]:


def create_pred_list_arimax(order, X_train, y_train, X_test, y_test, seasonal_order=(0, 0, 0, 0)):
    y_pred = pd.Series(index=y_test.index)
    
    for i in range(len(y_test)):
        temp_X_train = pd.concat([X_train, X_test[:i]])
        temp_X_test = X_test[i:]
        
        temp_y_train = pd.concat([y_train, y_test[:i]])
        temp_y_test = y_test[i:]
    
        model = SARIMAX(
            temp_y_train, 
            temp_X_train, 
            order=order, 
            seasonal_order=seasonal_order
        )
        model = model.fit()
        temp_y_pred = model.forecast(1, exog=temp_X_test[:1])
        
        y_pred[temp_y_pred.index] = temp_y_pred

    return y_pred


# In[10]:


def create_pred_list_ets(y_train, y_test, **kwargs):
    y_pred = pd.Series(index=y_test.index)

    for i in range(len(y_test)):
        temp_y_train = pd.concat([y_train, y_test[:i]])
        temp_y_test = y_test[i:]
    
        model = ETSModel(
            temp_y_train,
            error='add',
            trend='add',
            **kwargs,
        )
        model = model.fit()
        temp_y_pred = model.forecast(1)
        
        y_pred[temp_y_pred.index] = temp_y_pred

    return y_pred


# In[11]:


def create_pred_list_exp_smooth(y_train, y_test, **kwargs):
    y_pred = pd.Series(index=y_test.index)

    for i in range(len(y_test)):
        temp_y_train = pd.concat([y_train, y_test[:i]])
        temp_y_test = y_test[i:]
    
        model = ExponentialSmoothing(
            temp_y_train,
            trend='add',
            **kwargs,
        )
        model = model.fit()
        temp_y_pred = model.forecast(1)
        
        y_pred[temp_y_pred.index] = temp_y_pred

    return y_pred


# In[12]:


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    # в долях
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator)


# In[13]:


def weighted_mean_absolute_percentage_error(y_true, y_pred):
    # в долях
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    return numerator / denominator


# In[14]:


def show_metrics(y_true, y_pred):
    # MAPE не будем использовать, так как в среди истинных меток есть 0
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.3f}')
    print(f'RMSE: {root_mean_squared_error(y_true, y_pred):.3f}')
    print(f'SMAPE: {symmetric_mean_absolute_percentage_error(y_true, y_pred):.3f}')
    print(f'WMAPE: {weighted_mean_absolute_percentage_error(y_true, y_pred):.3f}')

    # график
    plt.title('Сравнение истинного значение с предсказанным')
    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='pred')
    plt.legend(loc='upper right')
    plt.show()


# ## Классические модели

# Так как раннее выяснили стационарность репрезитативного ряда, то явно это укажем при подборе параметров модели. Для ARIMA и её модификаций гиперпараметры будем подбирать автоматически 

# In[15]:


auto_arima_kwargs = dict(
    start_p=0, 
    max_p=5,
    start_q=0, 
    max_q=5,
    d=None,
    seasonal=False,
    stationary=True,
    trace=False,
    # random_state=random_state, раскомментировать, если поставим random=True
    n_fits=20,
    n_jobs=-1,
    scoring='mse', # можно поставить 'mae'
)


# In[16]:


auto_sarima_kwargs = dict(
    start_p=0, 
    max_p=5,
    start_q=0, 
    max_q=5,
    d=None,
    seasonal=True,
    stationary=True,
    trace=False,
    # random_state=random_state, раскомментировать, если поставим random=True
    n_fits=20,
    n_jobs=-1,
    scoring='mse', # можно поставить 'mae'
)


# ### ARIMA

# Подберем гиперпараметры для модели

# In[17]:


auto_model = auto_arima(
    y_train,
    **auto_arima_kwargs
)

print(auto_model.order)


# Попробуем предсказывать на одно значение вперед

# In[18]:


y_pred = create_pred_list_arima(auto_model.order, y_train, y_test)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[19]:


model = ARIMA(y_train, order=auto_model.order)
model = model.fit()
y_pred = model.forecast(len(y_test))
show_metrics(y_test, y_pred)


# ### ARIMAX

# К ARIMA добавляются дополнительные данные
# 
# Подберем гиперпараметры для модели

# In[20]:


auto_model = auto_arima(
    y_train,
    X_train,
    **auto_arima_kwargs
)

print(auto_model.order)


# Попробуем предсказывать на одно значение вперед

# In[21]:


y_pred = create_pred_list_arimax(auto_model.order, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[22]:


model = SARIMAX(y_train, X_train, order=auto_model.order)
model = model.fit()
y_pred = model.forecast(len(y_test), exog=X_test)
show_metrics(y_test, y_pred)


# ### SARIMA

# К ARIMA добавляется сезонность

# In[23]:


for m in [4, 7, 12]:
    auto_model = auto_arima(
        y_train,
        m=m,
        **auto_sarima_kwargs,
    )

    print(f'm = {m}, order = {auto_model.order}, seasonal_order = {auto_model.seasonal_order}')
    print('На одно значение вперед:')
    y_pred = create_pred_list_arima(auto_model.order, y_train, y_test, seasonal_order=auto_model.seasonal_order)
    show_metrics(y_test, y_pred)

    print('Вся тестовая выборка:')
    model = ARIMA(y_train, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    model = model.fit()
    y_pred = model.forecast(len(y_test))
    show_metrics(y_test, y_pred)


# ### SARIMAX

# К ARIMA добавляются дополнительные данные и сезонность

# In[24]:


for m in [4, 7, 12]:
    auto_model = auto_arima(
        y_train,
        X_train,
        m=m,
        **auto_sarima_kwargs,
    )

    print(f'm = {m}, order = {auto_model.order}, seasonal_order = {auto_model.seasonal_order}')
    print('На одно значение вперед:')
    y_pred = create_pred_list_arimax(auto_model.order, X_train, y_train, X_test, y_test, seasonal_order=auto_model.seasonal_order)
    show_metrics(y_test, y_pred)

    print('Вся тестовая выборка:')
    model = SARIMAX(y_train, exog=X_train, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    model = model.fit()
    y_pred = model.forecast(len(y_test), exog=X_test)
    show_metrics(y_test, y_pred)


# ### ETS

# Без сезонности
# 
# Попробуем предсказывать на одно значение вперед

# In[25]:


y_pred = create_pred_list_ets(y_train, y_test, seasonal=None)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[26]:


model = ETSModel(
    y_train,
    error='add',
    trend='add',
    seasonal=None,
)
    
model = model.fit()
y_pred = model.forecast(len(y_test))
show_metrics(y_test, y_pred)


# С сезонностью

# In[27]:


for seasonal_periods in [4, 7, 12]:   
    print(f'seasonal_periods = {seasonal_periods}')
    print('На одно значение вперед:')

    y_pred = create_pred_list_ets(y_train, y_test, seasonal='add', seasonal_periods=seasonal_periods)
    show_metrics(y_test, y_pred)


    print('Вся тестовая выборка:')
    model = ETSModel(
        y_train,
        error='add',
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods
    )
    model = model.fit()
    y_pred = model.forecast(len(y_test))
    show_metrics(y_test, y_pred)


# ### Holt-Winters

# Без сезонности
# 
# Попробуем предсказывать на одно значение вперед

# In[28]:


y_pred = create_pred_list_exp_smooth(y_train, y_test, seasonal=None)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[29]:


model = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal=None,
)
    
model = model.fit()
y_pred = model.forecast(len(y_test))
show_metrics(y_test, y_pred)


# С сезонностью

# In[30]:


for seasonal_periods in [4, 7, 12]:   
    print(f'seasonal_periods = {seasonal_periods}')
    print('На одно значение вперед:')

    y_pred = create_pred_list_exp_smooth(y_train, y_test, seasonal='add', seasonal_periods=seasonal_periods)
    show_metrics(y_test, y_pred)


    print('Вся тестовая выборка:')
    model = ExponentialSmoothing(
        y_train,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods
    )
    model = model.fit()
    y_pred = model.forecast(len(y_test))
    show_metrics(y_test, y_pred)


# По результатам проведенных экспериментов видно, что наилучшие значения метрик у модели ARIMAX с автоматически подобранными гиперпараметрами (вне зависимости от сезонности). Остальные модели отработали примерно как модель MovingAverage 

# In[ ]:




