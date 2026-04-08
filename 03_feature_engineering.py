#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


dataset_path = 'final_version_A.csv'
df = pd.read_csv(dataset_path)
df


# # ГЛАВА 3. FEATURE ENGINEERING ДЛЯ ВРЕМЕННЫХ РЯДОВ

# ## Лаговые признаки и скользящие статистики

# Лаговые признаки позволяют учитывать историю данных при прогнозирования. А скользящие статистики позволяют учитывать информацию о данных из определенного окна в прошлом.
# 
# Они отражают гипотезу о том, что текущее значение прогнозируемой величины зависит от её предыдущих значений.

# In[4]:


def get_lags_and_rollings(col_name, df):
    for lag in [1, 7, 14, 28]:
        new_col_name = f'lag_{lag}_{col_name}'
        df[new_col_name] = None
        for store_id in df['Store ID'].unique():
            for product_id in df['Product ID'].unique():
                temp_df = df.loc[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
                df[new_col_name].loc[temp_df.index] = temp_df[col_name].shift(lag)


        if lag != 1:
            new_col_name = f'mean_{lag}_{col_name}'
            df[new_col_name] = None
            for store_id in df['Store ID'].unique():
                for product_id in df['Product ID'].unique():
                    temp_df = df.loc[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
                    df[new_col_name].loc[temp_df.index] = temp_df[col_name].rolling(lag, closed='left').mean()

            
            new_col_name = f'std_{lag}_{col_name}'
            df[new_col_name] = None
            for store_id in df['Store ID'].unique():
                for product_id in df['Product ID'].unique():
                    temp_df = df.loc[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
                    df[new_col_name].loc[temp_df.index] = temp_df[col_name].rolling(lag, closed='left').std()


            new_col_name = f'max_{lag}_{col_name}'
            df[new_col_name] = None
            for store_id in df['Store ID'].unique():
                for product_id in df['Product ID'].unique():
                    temp_df = df.loc[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
                    df[new_col_name].loc[temp_df.index] = temp_df[col_name].rolling(lag, closed='left').max()
                    
            
            new_col_name = f'min_{lag}_{col_name}'
            df[new_col_name] = None
            for store_id in df['Store ID'].unique():
                for product_id in df['Product ID'].unique():
                    temp_df = df.loc[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
                    df[new_col_name].loc[temp_df.index] = temp_df[col_name].rolling(lag, closed='left').min()

    return df


# ### Лаги спроса

# В исходных данных за спрос отвечал признак "Demand Forecast", но в 01_eda он был удален

# ### Лаги заказов

# In[5]:


col_name = 'Units Ordered'

df = get_lags_and_rollings(col_name, df)
df


# ### Лаги скидок

# In[6]:


col_name = 'Discount'

df = get_lags_and_rollings(col_name, df)
df


# ### Лаги цен

# In[7]:


col_name = 'Price'

df = get_lags_and_rollings(col_name, df)
df


# ## Календарные признаки

# День недели, номер месяца и сезонность позволяют учитывать зависимость прогнозируемой величины от дня недели, номера месяца и сезона. Праздничный индикатор позволяет учитывать зависимость прогнозируемой величины от того, является ли день праздничным и выходным или нет.
# 
# Они отражают гипотезу о том, что разные дни, месяцы, сезоны влияют на прогнозируемую величину по-разному. Также иначе влияют праздничные дни

# ### День недели

# Реализовано в 01_eda

# ### Месяц

# Реализовано в 01_eda

# ### Праздничные индикаторы

# Реализовано в исходном датасете

# ### Сезонность

# Реализовано в исходном датасете

# ## Удаление NaN

# In[8]:


df.isna().sum().sort_values().tail(50)


# В столбце excess_category слишком много пропущенных значений, поэтому удалим его

# In[9]:


df = df.drop('excess_category', axis=1)


# Удалим строки с пропущенными значениями, так как они образовались после генерации новых признаков, а их удаление не испортит сортировку по датам

# In[10]:


df = df.dropna()
df


# In[11]:


df.isna().sum().sort_values().tail(5)


# ## Сохранение очищенного датасета

# In[12]:


df.to_csv('clean_dataset.csv', index=0)


# Проверим корректность сохранения

# In[13]:


pd.read_csv('clean_dataset.csv')


# ## Предобработка численных и категориальных данных

# Так как предсказания будут строиться по определенному репрезентативному ряду, то для примера возьмем данные с Store ID = S001 и Product ID = P0001

# In[14]:


store_id = 'S001'
product_id = 'P0001'
temp_df = df.set_index(['Store ID', 'Product ID', 'Date']).loc[store_id, product_id]
temp_df


# ### Разделение на тренировочную и тестовую выборки

# Разделим репрезентативный ряд в отношении 80 на 20 (примерно)

# In[15]:


threshold = int(len(temp_df)*0.8)
target = 'Units Sold'
y = temp_df[target]
X = temp_df.drop(target, axis=1)

X_train = X[:threshold]
X_test = X[threshold:]

y_train = y[:threshold]
y_test = y[threshold:]


# In[16]:


X_train


# In[17]:


X_test


# ### Разделение на численные и категориальные данные

# In[18]:


cat_cols = ['Category', 'Region', 
           'Discount', 'Weather Condition',
           'Holiday/Promotion', 'Seasonality',
           'day_of_week', 'month']


# In[19]:


for col in cat_cols:
    print(temp_df[col].unique())


# In[20]:


num_cols = sorted(list(set(X_train.columns) - set(cat_cols)))


# ### Трансформирование данных

# In[21]:


ohe = OneHotEncoder(drop='first')
X_train_cat = pd.DataFrame(ohe.fit_transform(X_train[cat_cols], y_train).todense(), columns=ohe.get_feature_names_out(), index=X_train.index)
X_test_cat = pd.DataFrame(ohe.transform(X_test[cat_cols]).todense(), columns=ohe.get_feature_names_out(), index=X_test.index)


# In[22]:


scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols], y_train), columns=scaler.get_feature_names_out(), index=X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=scaler.get_feature_names_out(), index=X_test.index)


# In[23]:


X_train_cat


# In[24]:


X_train_transformed = pd.merge(X_train_cat, X_train_num, on='Date')
X_test_transformed = pd.merge(X_test_cat, X_test_num, on='Date')


# In[25]:


train_transformed = pd.merge(X_train_transformed, y_train, on='Date')
test_transformed = pd.merge(X_test_transformed, y_test, on='Date')


# In[26]:


train_transformed


# In[27]:


test_transformed


# ### Сохранение полученных датасетов

# In[28]:


train_transformed.to_csv('train.csv')

pd.read_csv('train.csv', index_col='Date')


# In[29]:


test_transformed.to_csv('test.csv')

pd.read_csv('test.csv', index_col='Date')

