#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install catboost')


# In[2]:


import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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


# # ГЛАВА 6. ENSEMBLE-МОДЕЛИ

# In[8]:


def create_pred_list(model_class, X_train, y_train, X_test, y_test, **kwargs):
    y_pred = pd.Series(index=y_test.index)

    for i in range(len(y_test)):
        temp_X_train = pd.concat([X_train, X_test[:i]])
        temp_X_test = X_test[i:]

        temp_y_train = pd.concat([y_train, y_test[:i]])
        temp_y_test = y_test[i:]

        model = model_class(**kwargs)
        model.fit(temp_X_train, temp_y_train)

        temp_y_pred = model.predict(temp_X_test[:1])

        y_pred[temp_y_test.index[0]] = temp_y_pred

    return y_pred


# In[9]:


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator)


# In[10]:


def show_metrics(y_true, y_pred):
    # MAPE не будем использовать, так как среди истинных меток есть 0
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.3f}')
    print(f'RMSE: {root_mean_squared_error(y_true, y_pred):.3f}')
    print(f'SMAPE: {symmetric_mean_absolute_percentage_error(y_true, y_pred):.3f}')

    # график
    plt.title('Сравнение истинного значение с предсказанным')
    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='pred')
    plt.legend(loc='upper right')
    plt.show()


# ## ML-модели для временных рядов

# Будем подбирать гиперпараметры для предсказания всей выборки, а не следующего события

# ### RandomForestRegressor

# In[11]:


param_grid = {
    'n_estimators': [100, 200],
    # 'criterion': ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'],
    # 'max_depth': list(range(3, 10, 2)) + [None],
    # 'min_samples_split': range(2, 5),
    # 'min_samples_leaf': range(1, 4),
    # 'max_features': ['sqrt', 'log2', None],
    # 'bootstrap': [True, False],
    'random_state': [random_state],
    'n_jobs': [-1],
}

model_class = RandomForestRegressor


# In[12]:


start_time = time.time()

grid_search = GridSearchCV(
    estimator=model_class(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', # neg_ так как ищутся параметры для модели с наибольшим значением метрики
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}\nзаняло: {time.time() - start_time} секунд")


# Попробуем предсказывать на одно значение вперед

# In[13]:


y_pred = create_pred_list(model_class, X_train, y_train, X_test, y_test, **grid_search.best_params_)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[14]:


model = model_class(**grid_search.best_params_)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)
show_metrics(y_test, y_pred)


# ### GradientBoostingRegressor

# In[15]:


param_grid = {
    # 'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    # 'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [100, 200],
    # 'criterion': ['friedman_mse', 'squared_error'],
    # 'min_samples_split': range(2, 5),
    # 'min_samples_leaf': range(1, 4),
    # 'max_depth': list(range(3, 6, 2)) + [None],
    # 'max_features': ['sqrt', 'log2', None],
    'random_state': [random_state],
}

model_class = GradientBoostingRegressor


# In[16]:


start_time = time.time()

grid_search = GridSearchCV(
    estimator=model_class(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', # neg_ так как ищутся параметры для модели с наибольшим значением метрики
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}\nзаняло: {time.time() - start_time} секунд")


# Попробуем предсказывать на одно значение вперед

# In[17]:


y_pred = create_pred_list(model_class, X_train, y_train, X_test, y_test, **grid_search.best_params_)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[18]:


model = model_class(**grid_search.best_params_)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)
show_metrics(y_test, y_pred)


# ### LGBMRegressor

# In[26]:


param_grid = {
    # 'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [100, 200],
    # 'min_split_gain': range(2, 5),
    # 'min_child_samples': range(1, 4),
    # 'max_depth': list(range(3, 6, 2)) + [None],
    'random_state': [random_state],
    'verbosity': [-1],
}

model_class = LGBMRegressor


# In[27]:


start_time = time.time()

grid_search = GridSearchCV(
    estimator=model_class(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', # neg_ так как ищутся параметры для модели с наибольшим значением метрики
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}\nзаняло: {time.time() - start_time} секунд")


# Попробуем предсказывать на одно значение вперед

# In[28]:


y_pred = create_pred_list(model_class, X_train, y_train, X_test, y_test, **grid_search.best_params_)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[29]:


model = model_class(**grid_search.best_params_)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)
show_metrics(y_test, y_pred)


# ### XGBRegressor

# In[30]:


param_grid = {
    # 'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [100, 200],
    # 'max_depth': list(range(3, 6, 2)) + [None],
    'random_state': [random_state],
    # 'booster': ['gbtree', 'gblinear', 'dart'],
    'n_jobs': [-1],
}

model_class = XGBRegressor


# In[31]:


start_time = time.time()

grid_search = GridSearchCV(
    estimator=model_class(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', # neg_ так как ищутся параметры для модели с наибольшим значением метрики
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}\nзаняло: {time.time() - start_time} секунд")


# Попробуем предсказывать на одно значение вперед

# In[32]:


y_pred = create_pred_list(model_class, X_train, y_train, X_test, y_test, **grid_search.best_params_)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[33]:


model = model_class(**grid_search.best_params_)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)
show_metrics(y_test, y_pred)


# ### CatBoostRegressor

# In[34]:


param_grid = {
    # 'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [100, 200],
    # 'max_depth': list(range(3, 6, 2)) + [None],
    'random_state': [random_state],
    'logging_level': ['Silent'],
}

model_class = CatBoostRegressor


# In[35]:


start_time = time.time()

grid_search = GridSearchCV(
    estimator=model_class(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', # neg_ так как ищутся параметры для модели с наибольшим значением метрики
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}\nзаняло: {time.time() - start_time} секунд")


# Попробуем предсказывать на одно значение вперед

# In[36]:


y_pred = create_pred_list(model_class, X_train, y_train, X_test, y_test, **grid_search.best_params_)
show_metrics(y_test, y_pred)


# Попробуем предсказывать всю тестовую выборку

# In[37]:


model = model_class(**grid_search.best_params_)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)
show_metrics(y_test, y_pred)


# ## Ансамблирование

# Все рассмотренные выше модели являются ансамблями

# Лучшие значения метрик наблюдаются у GradientBoostingRegressor. У всех остальных рассмотренных моделей значиения метрик лучше, чем у наивных моделей и статистических
