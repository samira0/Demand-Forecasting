#!/usr/bin/env python
# coding: utf-8

# # Baseline Models

# In[1]:


import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


df_train = pd.read_csv('train.csv', index_col='Date')
df_train.index = pd.to_datetime(df_train.index)


# In[4]:


df_test = pd.read_csv('test.csv', index_col='Date')
df_test.index = pd.to_datetime(df_test.index)


# In[5]:


target = 'Units Sold'

y_train = df_train[target]
X_train = df_train.drop(target, axis=1)

y_test = df_test[target]
X_test = df_test.drop(target, axis=1)


# In[6]:


def create_pred_list(model, X_train, y_train, X_test, y_test):
    y_pred = pd.Series(index=y_test.index)

    for i in range(len(y_test)):
        temp_X_train = pd.concat([X_train, X_test[:i]])
        temp_X_test = X_test[i:]
    
        temp_y_train = pd.concat([y_train, y_test[:i]])
        temp_y_test = y_test[i:]
    
    
        model.fit(temp_X_train, temp_y_train)
        temp_y_pred = model.predict(temp_X_test)
        
        y_pred[temp_y_pred.index] = temp_y_pred

    return y_pred


# In[7]:


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator)


# In[8]:


def show_metrics(y_true, y_pred):
    # We will not use MAPE, since among the true labels there are 0
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.3f}')
    print(f'RMSE: {root_mean_squared_error(y_true, y_pred):.3f}')
    print(f'SMAPE: {symmetric_mean_absolute_percentage_error(y_true, y_pred):.3f}')

    # Plot
    plt.title('Comparison of the true value with the predicted value')
    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='pred')
    plt.legend(loc='upper right')
    plt.show()


# ## Naive (last value)

# In[9]:


class Naive(RegressorMixin):
    """
    Since we only know the values in the training set, 
    we can only predict one step ahead (the last value from the training set). 
    If we receive multiple examples, the predictions for them 
    will be the same—the last value from the training set.
    """
    def __init__(self, *params):
        ...
        
    def fit(self, X, y):
        self.last_date = max(y.index)
        self.last_value = y[self.last_date]
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        y_pred = pd.Series(self.last_value, index=X.index)
        return y_pred


# In[10]:


model = Naive()

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[11]:


model = Naive()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# ## Seasonal Naive

# In[12]:


class SeasonalNaive(RegressorMixin):
    def __init__(self, seas_lag, *params):
        self.seas_lag = seas_lag
        
    def fit(self, X, y):
        y_sorted = y.sort_index()
        self.last_values = y_sorted[-self.seas_lag:].values
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        y_pred = pd.Series(index=X.index)
        for i in range(len(y_pred)):
            i_seas = i % self.seas_lag
            y_pred.iloc[i] = self.last_values[i_seas]
        return y_pred


# In[13]:


model = SeasonalNaive(7)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[14]:


model = SeasonalNaive(7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[15]:


model = SeasonalNaive(30)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[16]:


model = SeasonalNaive(30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# ## Moving Average

# In[17]:


class MovingAverage(RegressorMixin):
    def __init__(self, window_size, *params):
        self.window_size = window_size
        
    def fit(self, X, y):
        y_sorted = y.sort_index()
        self.last_values = y_sorted[-self.window_size:].values
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        y_pred = pd.Series(index=X.index)
        for i in range(len(y_pred)):
            y_pred.iloc[i] = np.mean(self.last_values)
        return y_pred


# In[18]:


model = MovingAverage(3)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[19]:


model = MovingAverage(3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[20]:


model = MovingAverage(7)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[21]:


model = MovingAverage(7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[22]:


model = MovingAverage(14)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[23]:


model = MovingAverage(14)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[24]:


model = MovingAverage(28)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[25]:


model = MovingAverage(28)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[26]:


model = MovingAverage(30)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[27]:


model = MovingAverage(30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[28]:


model = MovingAverage(60)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[29]:


model = MovingAverage(60)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)


# In[30]:


model = MovingAverage(90)

y_pred = create_pred_list(model, X_train, y_train, X_test, y_test)
show_metrics(y_test, y_pred)


# Predicting the entire test sample

# In[31]:


model = MovingAverage(90)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
show_metrics(y_test, y_pred)

