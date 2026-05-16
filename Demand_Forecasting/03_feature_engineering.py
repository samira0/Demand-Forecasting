#!/usr/bin/env python
# coding: utf-8

# # FEATURE ENGINEERING

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('data_enriched.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(
    ['Store ID', 'Product ID', 'Date']
).reset_index(drop=True)


# ## Lags and Rollings

# Lagged features allow historical data to be taken into account when forecasting. Statistics reduction allows for information to be obtained from data within a specific window in the past.
# 
# They reflect the hypothesis that the current value of the predicted variable depends on its previous measurements.

# In[5]:


df['excess_sales'] = df['Units Sold'] - df['Units Ordered']

print(f"excess_sales added:")
print(f"  Mean: {df['excess_sales'].mean():.3f}  ")
print(f"  Max:  {df['excess_sales'].max():.3f}")
print(f"  Zero values: {(df['excess_sales'] == 0).sum()} "
      f"({(df['excess_sales']==0).mean()*100:.1f}%)")


# In[6]:


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


# In[7]:


df = get_lags_and_rollings('Units Sold', df)
df = get_lags_and_rollings('Units Ordered', df)
df = get_lags_and_rollings('excess_sales', df)
df = get_lags_and_rollings('Price', df)
df = get_lags_and_rollings('Discount', df)


# ## Categoricals Encoding and Series Preparation

# In[8]:


for col in ['temp_mean', 'temp_min', 'temp_max',
            'precipitation', 'wind_speed', 'sunshine',
            'is_cold', 'heavy_rain']:
    new_col = f'lag_1_{col}'
    df[new_col] = None
    for store_id in df['Store ID'].unique():
        for product_id in df['Product ID'].unique():
            temp_df = df.loc[
                (df['Store ID'] == store_id) &
                (df['Product ID'] == product_id)
            ]
            df.loc[temp_df.index, new_col] = temp_df[col].shift(1)

for col in ['Category', 'Region', 'Seasonality']:
    df[f'{col}_enc'] = df[col].astype('category').cat.codes

print(f"Shape after feature engineering: {df.shape}")

series = df[
    (df['Store ID'] == 'S001') &
    (df['Product ID'] == 'P0001')
].copy().sort_values('Date').reset_index(drop=True)

# Drop warmup rows
series = series.dropna().reset_index(drop=True)
print(f"Single series after dropna: {len(series)} rows")

FEATURE_COLS = [
    # Core
    'Units Ordered',
    'excess_sales',                               
    'Price', 'Discount',
    'Holiday/Promotion', 'Competitor Pricing',
    # Current weather
    'temp_mean', 'temp_max', 'temp_min',
    'precipitation', 'wind_speed', 'sunshine',
    # Holiday flags
    'real_holiday', 'pre_holiday', 'post_holiday', 'holiday_week',
    # Calendar
    'day_of_week', 'month', 'quarter', 'week_of_year',
    'is_weekend', 'day_of_month', 'is_month_start', 'is_month_end',
    # Weather interactions
    'is_cold', 'heavy_rain', 'cold_holiday',
    # Lag features
    'lag_1_Units Sold', 'lag_1_Units Ordered',
    'lag_1_excess_sales',                         
    'lag_1_Price', 'lag_1_Discount',
    'lag_7_Units Sold', 'lag_7_Units Ordered',
    'lag_7_excess_sales',                         
    'lag_7_Price', 'lag_7_Discount',
    'lag_14_Units Sold', 'lag_14_Units Ordered',
    'lag_14_excess_sales',                        
    'lag_14_Price', 'lag_14_Discount',
    'lag_28_Units Sold', 'lag_28_Units Ordered',
    'lag_28_excess_sales',                        
    'lag_28_Price', 'lag_28_Discount',
    # Rolling statistics
    'mean_7_Units Sold',  'std_7_Units Sold',
    'max_7_Units Sold',   'min_7_Units Sold',
    'mean_7_Units Ordered',
    'mean_7_excess_sales',                        
    'mean_14_Units Sold', 'std_14_Units Sold',
    'max_14_Units Sold',  'min_14_Units Sold',
    'mean_14_Units Ordered',
    'mean_14_excess_sales',                       
    'mean_28_Units Sold', 'std_28_Units Sold',
    'max_28_Units Sold',  'min_28_Units Sold',
    'mean_28_Units Ordered',
    'mean_28_excess_sales',                       
    # Lagged weather
    'lag_1_temp_mean', 'lag_1_temp_min', 'lag_1_temp_max',
    'lag_1_precipitation', 'lag_1_wind_speed', 'lag_1_sunshine',
    'lag_1_is_cold', 'lag_1_heavy_rain',
    # Encoded categoricals
    'Category_enc', 'Region_enc', 'Seasonality_enc',
]

TARGET = 'Units Sold'

# Check all feature columns exist
missing = [c for c in FEATURE_COLS if c not in series.columns]
if missing:
    print(f"WARNING — missing columns: {missing}")
else:
    print("All feature columns present.")


# ## Train / Test Split

# In[9]:


cutoff = int(len(series) * 0.8)
train  = series.iloc[:cutoff].copy()
test   = series.iloc[cutoff:].copy()

print(f"Train: {len(train)} rows | Test: {len(test)} rows")
print(f"Train period: {train['Date'].min()} to {train['Date'].max()}")
print(f"Test period:  {test['Date'].min()} to {test['Date'].max()}")


# In[10]:


scaler   = StandardScaler()
num_cols = [c for c in FEATURE_COLS
            if series[c].dtype in [np.float64, np.int64, float, int]]

train_out = train[FEATURE_COLS + [TARGET, 'Date']].copy()
test_out  = test[FEATURE_COLS  + [TARGET, 'Date']].copy()

train_out[num_cols] = scaler.fit_transform(train[num_cols])
test_out[num_cols]  = scaler.transform(test[num_cols])

train_out = train_out.set_index('Date')
test_out  = test_out.set_index('Date')

train_out.to_csv('train.csv')
test_out.to_csv('test.csv')

# Full enriched panel for multi-series scripts
df.to_csv('data_enriched_with_features.csv', index=False)

