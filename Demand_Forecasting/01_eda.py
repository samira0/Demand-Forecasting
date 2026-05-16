#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Analysis
# 
# 

# ## Dataset description

# In[1]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


TARGET_COLUMN = "Units Sold"
ID_COLUMNS = ["Store ID", "Product ID"]
DATE_COLUMN = "Date"
KEY_COLUMNS = ["Date", "Store ID", "Product ID"]
LEAKAGE_COLUMNS = ["Demand Forecast", "Inventory Level"]

CATEGORICAL_COLUMNS = [
    "Store ID", "Product ID", "Category", "Region",
    "Weather Condition", "Seasonality",
]
NUMERICAL_COLUMNS = [
    "Units Sold", "Units Ordered", "Price", "Discount",
    "Holiday/Promotion", "Competitor Pricing",
]

df = pd.read_csv('retail_store_inventory.csv')


# ## Structure and data type

# In[3]:


print(f"Data shape: {df.shape[0]} строк, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nData type info and missing values:")
df.info()


# In[4]:


print(f"1. Target column: '{TARGET_COLUMN}'")
print("   - Number of units sold")
print("   - Directly reflects consumer demand")
print("   - Metric for assessing forecast quality")
print(f"\n2. ID columns: {ID_COLUMNS}")
print("   - Used only for grouping data")
print(f"\n3. Date column: '{DATE_COLUMN}'")
print("   - Defines a temporary data structure")
print(f"\n4. Categorical Features: {CATEGORICAL_COLUMNS} ({len(CATEGORICAL_COLUMNS)} шт.)")
print(f"\n5. Numeric features: {NUMERICAL_COLUMNS} ({len(NUMERICAL_COLUMNS)} шт.)")


# ## Data type conversion

# In[5]:


df["Date"] = pd.to_datetime(df["Date"])
for col in CATEGORICAL_COLUMNS:
    if col in df.columns:
        df[col] = df[col].astype("category")
print("\nChecking converted data types:")
print(df.dtypes.to_string())


# ## Analysis and handling of missing values, duplicates and target variable

# In[6]:


missing_values = df.isna().sum()
print(f"Number of missing values:\n{missing_values[missing_values > 0] if missing_values.any() else 'No missing values'}")


# In[7]:


full_duplicates = df.duplicated().sum()
print(f"Number of complete duplicate rows: {full_duplicates}")
date_duplicates = df.duplicated(subset=KEY_COLUMNS).sum()
print(f"Number of duplicates by key fields {KEY_COLUMNS}: {date_duplicates}")


# In[8]:


target_stats = df[TARGET_COLUMN].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print("Target variable statistics")
print(target_stats.to_string())
mean_sold = df[TARGET_COLUMN].mean()
print(f"\nAdditional metrics:")
print(f"Coefficient of variation: {df[TARGET_COLUMN].std() / mean_sold * 100:.2f}%" if mean_sold else "N/A")
print(f"Asymmetry: {df[TARGET_COLUMN].skew():.4f}")
print(f"Excess: {df[TARGET_COLUMN].kurtosis():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df[TARGET_COLUMN], bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Units Sold")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram of the distribution of Units Sold")
axes[0].grid(True, alpha=0.3)
axes[1].boxplot(df[TARGET_COLUMN], vert=False)
axes[1].set_xlabel("Units Sold")
axes[1].set_title("Boxplot Units Sold")
stats.probplot(df[TARGET_COLUMN], dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot (comparison with normal distribution)")
plt.tight_layout()
plt.show()

zero_sales = (df[TARGET_COLUMN] == 0).sum()
print(f"Days with zero sales: {zero_sales} ({zero_sales/len(df)*100:.2f}%)")
print(f"Maximum value: {df[TARGET_COLUMN].max()}")


# ## Checking the logical consistency of data

# In[9]:


price_issues = {
    "negative_prices": (df["Price"] <= 0).sum(),
    "negative_competitor_prices": (df["Competitor Pricing"] <= 0).sum(),
}
print("Checking price features:")
for issue, count in price_issues.items():
    print(f"   {issue}: {count} rows")

print("\nComparison of Units Sold and Units Ordered:")
sold_exceeds_ordered = (df["Units Sold"] > df["Units Ordered"]).sum()
print(f"Sales exceed orders in {sold_exceeds_ordered} cases ({sold_exceeds_ordered/len(df)*100:.2f}%)")

if sold_exceeds_ordered > 0:
    excess_cases = df[df["Units Sold"] > df["Units Ordered"]]
    diff = excess_cases["Units Sold"] - excess_cases["Units Ordered"]
    print(f"\n   Average excess: {diff.mean():.2f}")
    print(f"   Maximum excess: {diff.max()}")


# ## A detailed analysis of the 51.4% overselling problem

# In[10]:


df["excess_sales"] = df["Units Sold"] - df["Units Ordered"]

print("\n1. EXCESS STATISTICS:")
excess_positive = df.loc[df["excess_sales"] > 0, "excess_sales"]
print(excess_positive.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

print("\n2. DISTRIBUTION BY EXCESS AMOUNT:")
bins = [0, 10, 50, 100, 200, 500]
labels = ["0-10", "11-50", "51-100", "101-200", ">200"]
excess_binned = pd.cut(excess_positive, bins=bins, labels=labels)
excess_distribution = excess_binned.value_counts().sort_index()
for category, count in excess_distribution.items():
    print(f"  Excess {category}: {count} cases ({count / sold_exceeds_ordered * 100:.1f}%)")

print("\n3. ANALYSIS BY STORES (TOP 10 by number of anomalies):")
store_issues = df[df["excess_sales"] > 0].groupby("Store ID").size().sort_values(ascending=False).head(10)
for store, count in store_issues.items():
    store_total = (df["Store ID"] == store).sum()
    print(f"  Store {store}: {count} anomalies from {store_total} records ({count/store_total*100:.1f}%)")

print("\n4. PRODUCT ANALYSIS (TOP 10):")
product_issues = df[df["excess_sales"] > 0].groupby("Product ID").size().sort_values(ascending=False).head(10)
for product, count in product_issues.items():
    product_total = (df["Product ID"] == product).sum()
    print(f"  Product {product}: {count} anomalies from {product_total} records ({count/product_total*100:.1f}%)")

print("\n5. TIME ANALYSIS:")
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("  By day of the week:")
for day in range(7):
    mask = df["day_of_week"] == day
    anomalies = (df.loc[mask, "Units Sold"] > df.loc[mask, "Units Ordered"]).sum()
    total = mask.sum()
    pct = anomalies / total * 100 if total > 0 else 0
    print(f"    {day_names[day]}: {anomalies}/{total} ({pct:.1f}%)")

print("\n6. CORRELATION WITH OTHER FEATURES:")
discount_anomalies = ((df["excess_sales"] > 0) & (df["Discount"] > 0)).sum()
print(f"  Anomalies at a discount: {discount_anomalies} ({discount_anomalies/sold_exceeds_ordered*100:.1f}%)")
holiday_anomalies = ((df["excess_sales"] > 0) & (df["Holiday/Promotion"] == 1)).sum()
print(f"  Anomalies during holidays: {holiday_anomalies} ({holiday_anomalies/sold_exceeds_ordered*100:.1f}%)")


# ### Visualization of the problem

# In[11]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
excess_only = df.loc[df["excess_sales"] > 0, "excess_sales"]
axes[0, 0].hist(excess_only, bins=50, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Excess of sales over orders")
axes[0, 0].set_ylabel("Number of cases")
axes[0, 0].set_title("Distribution of the excess value")
axes[0, 0].axvline(x=100, color="r", linestyle="--", alpha=0.5, label="Threshold 100")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

top_stores = df[df["excess_sales"] > 0].groupby("Store ID").size().nlargest(5).index
store_data = df[df["Store ID"].isin(top_stores) & (df["excess_sales"] > 0)]
sns.boxplot(data=store_data, x="Store ID", y="excess_sales", ax=axes[0, 1])
axes[0, 1].set_title("Distribution of excess by store (top 5)")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].set_ylabel("Excess")

pivot_table = df.pivot_table(
    values="excess_sales",
    index="month",
    columns="day_of_week",
    aggfunc=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0,
)
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0, 2])
axes[0, 2].set_title("% of anomalies by month and day of the week")
axes[0, 2].set_xlabel("Day of the week (0=Mon)")
axes[0, 2].set_ylabel("Month")

by_date = df.set_index("Date")["excess_sales"].gt(0).astype(int)
weekly_pct = by_date.resample("W").mean() * 100
axes[1, 0].plot(weekly_pct.index, weekly_pct.values)
axes[1, 0].set_xlabel("Date")
axes[1, 0].set_ylabel("% of abnormal records")
axes[1, 0].set_title("Dynamics of anomalies (weekly)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=50, color="r", linestyle="--", alpha=0.5, label="50%")

discount_bins = pd.cut(df["Discount"], bins=[-1, 0, 5, 10, 15, 20])
discount_pct = df.groupby(discount_bins).apply(
    lambda g: (g["Units Sold"] > g["Units Ordered"]).sum() / len(g) * 100
)
discount_pct.plot(kind="bar", ax=axes[1, 1])
axes[1, 1].set_xlabel("Discount level")
axes[1, 1].set_ylabel("% of anomalies")
axes[1, 1].set_title("The impact of discounts on anomalies")
axes[1, 1].tick_params(axis="x", rotation=45)

axes[1, 2].scatter(df["Units Ordered"], df["Units Sold"], alpha=0.3, s=10)
max_ord = df["Units Ordered"].max()
axes[1, 2].plot([0, max_ord], [0, max_ord], "r--", label="y=x (sales=orders)")
axes[1, 2].set_xlabel("Units Ordered")
axes[1, 2].set_ylabel("Units Sold")
axes[1, 2].set_title("Sales vs Orders")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(f"Anomaly analysis: sales > orders ({sold_exceeds_ordered/len(df)*100:.2f}% cases)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# ### Methodological solution of the problem

# In[12]:


print("\nANALYSIS OF PROCESSING OPTIONS:")

# Option 1: Complete exclusion of the feature
print("\n1. COMPLETE EXCLUSION of Units Ordered:")
print("   Pros:")
print("   - Eliminates the problem completely")
print("   - Simplifies the model")
print("   Cons:")
print("   - Losing potentially useful information")
print("   - Units Ordered may contain forward-looking information")

# Option 2: Data correction
print("\n2. DATA CORRECTION (Units_Ordered_Corrected = max(sales, orders)):")
print("   Pros:")
print("   - Logically correct data")
print("   - Save information about orders")
print("   - A simple and clear solution")
print("   Cons:")
print("   - Changes the original data")
print("   - May distort order statistics")

# Option 3: Two versions of data
print("\n3. TWO VERSIONS of DATA (with and without correction):")
print("   Pros:")
print("   - Scientific rigor (comparison of approaches)")
print("   - Allows to evaluate the impact of the correction")
print("   Cons:")
print("   - Complicates analysis")


# ### Creating two versions of data

# In[13]:


EDA_ONLY_COLUMNS = ["excess_sales", "day_of_week", "month"]

# Version A: correct Units Ordered (max(sold, ordered)), then drop temp and leakage
df_version_a = df.drop(columns=EDA_ONLY_COLUMNS + LEAKAGE_COLUMNS, errors="ignore").copy()
corrected = np.maximum(df_version_a["Units Ordered"], df_version_a["Units Sold"])
df_version_a["Units Ordered"] = corrected

# Version B: no correction; drop EDA-only and leakage
df_version_b = df.drop(columns=EDA_ONLY_COLUMNS + LEAKAGE_COLUMNS, errors="ignore").copy()

# Working dataframe for downstream: use version B (no target leakage)
df = df_version_b.copy()

# Comparison table
print("STATISTICAL COMPARISON OF TWO VERSIONS")
comparison_stats = pd.DataFrame(
    {
        "Recordings with anomalies": [
            (df_version_a["Units Sold"] > df_version_a["Units Ordered"]).sum(),  # 0 after correction
            (df_version_b["Units Sold"] > df_version_b["Units Ordered"]).sum(),
        ],
        "Mean Units Ordered": [
            df_version_a["Units Ordered"].mean(),
            df_version_b["Units Ordered"].mean(),
        ],
        "Standard deviation": [
            df_version_a["Units Ordered"].std(),
            df_version_b["Units Ordered"].std(),
        ],
        "Median": [
            df_version_a["Units Ordered"].median(),
            df_version_b["Units Ordered"].median(),
        ],
    },
    index=["Version A (with correction)", "Version B (without correction)"],
)
print(comparison_stats.round(2).to_string())

print("\nVERSION A (Units Ordered = max(Ordered, Sold)):")
print(df_version_a[["Units Sold", "Units Ordered"]].describe().to_string())
print("\nVERSION B (without correction):")
print(df_version_b[["Units Sold", "Units Ordered"]].describe().to_string())


def save_final_versions():
    """Write final_version_A.csv and final_version_B.csv to DATA_DIR."""
    df_version_a.to_csv(OUTPUT_VERSION_A, index=False)
    df_version_b.to_csv(OUTPUT_VERSION_B, index=False)
    print(f"  {OUTPUT_VERSION_A.name}  — version with anomaly correction")
    print(f"  {OUTPUT_VERSION_B.name}  — version without correction")


# In[14]:


from scipy import stats

df_version_a = df.copy()
df_version_b = df.copy()

print("\nVERSION A (WITH CORRECTION):")
print("Units_Ordered_Corrected = np.maximum(Units Ordered, Units Sold)")
df_version_a['Units_Ordered_Corrected'] = np.maximum(df_version_a['Units Ordered'], df_version_a['Units Sold'])
df_version_a['Was_Corrected'] = (df_version_a['Units_Ordered_Corrected'] != df_version_a['Units Ordered']).astype(int)

print("\nVERSION B (WITHOUT CORRECTION):")
print("Keep the original data unchanged")
df_version_b['Units_Ordered_Original'] = df_version_b['Units Ordered'].copy()
df_version_b['Has_Anomaly'] = (df_version_b['Units Sold'] > df_version_b['Units Ordered']).astype(int)

print("STATISTICAL COMPARISON OF TWO VERSIONS")

comparison_stats = pd.DataFrame(index=['Version A (with correction)', 'Version B (without correction)'])

# Basic Statistics
comparison_stats['Recordings with anomalies'] = [
    df_version_a['Was_Corrected'].sum(),
    df_version_b['Has_Anomaly'].sum()
]

comparison_stats['% of anomalies'] = [
    df_version_a['Was_Corrected'].mean() * 100,
    df_version_b['Has_Anomaly'].mean() * 100
]

comparison_stats['Mean Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].mean(),
    df_version_b['Units Ordered'].mean()
]

comparison_stats['Standard deviation Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].std(),
    df_version_b['Units Ordered'].std()
]

comparison_stats['Median Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].median(),
    df_version_b['Units Ordered'].median()
]

comparison_stats['Min Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].min(),
    df_version_b['Units Ordered'].min()
]

comparison_stats['Max Units Ordered'] = [
    df_version_a['Units_Ordered_Corrected'].max(),
    df_version_b['Units Ordered'].max()
]

print(comparison_stats.round(2).to_string())


# In[15]:


df_version_a = df.copy()
df_version_b = df.copy()


# In[16]:


df_version_a['Units_Ordered_Corrected'] = np.maximum(
    df_version_a['Units Ordered'],
    df_version_a['Units Sold']
)

# replace Units Ordered with the adjusted one
df_version_a['Units Ordered'] = df_version_a['Units_Ordered_Corrected']
df_version_a = df_version_a.drop(columns=['Units_Ordered_Corrected'])


# In[17]:


#Eliminating features with information leakage
leakage_cols = ['Demand Forecast', 'Inventory Level']

df_version_a = df_version_a.drop(columns=leakage_cols, errors='ignore')
df_version_b = df_version_b.drop(columns=leakage_cols, errors='ignore')


# Demand Forecast is a pre-calculated demand forecast.
# 
# Inventory Level reflects supply constraints, not consumer demand.

# In[18]:


print("VERSION A:")
print(df_version_a[['Units Sold','Units Ordered']].describe())

print("\nVERSION B (without corrections):")
print(df_version_b[['Units Sold','Units Ordered']].describe())


# In[19]:


df_version_a.to_csv("final_version_A.csv", index=False)

import requests

df = pd.read_csv('final_version_A.csv')
df['Date'] = pd.to_datetime(df['Date'])
print("Loaded Version A:", df.shape)
print("Regions:", df['Region'].unique())

# Region → Russian city coordinates
REGION_COORDS = {
    'North': {
        'city': 'Saint Petersburg',
        'lat': 59.95, 'lon': 30.32
    },
    'South': {
        'city': 'Rostov-on-Don',
        'lat': 47.23, 'lon': 39.72
    },
    'East': {
        'city': 'Novosibirsk',
        'lat': 54.99, 'lon': 82.90
    },
    'West': {
        'city': 'Moscow',
        'lat': 55.75, 'lon': 37.62
    },
}

# Fetch real daily weather from Open-Meteo
def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ["temperature_2m_mean", "temperature_2m_max",
                  "temperature_2m_min", "precipitation_sum",
                  "windspeed_10m_max", "shortwave_radiation_sum"],
        "timezone": "Europe/Moscow"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        return None
    data = r.json()
    w = pd.DataFrame(data['daily'])
    w['time'] = pd.to_datetime(w['time'])
    w = w.rename(columns={
        'time': 'Date', 'temperature_2m_mean': 'temp_mean',
        'temperature_2m_max': 'temp_max', 'temperature_2m_min': 'temp_min',
        'precipitation_sum': 'precipitation', 'windspeed_10m_max': 'wind_speed',
        'shortwave_radiation_sum': 'sunshine'
    })
    return w

start_str = df['Date'].min().strftime('%Y-%m-%d')
end_str   = df['Date'].max().strftime('%Y-%m-%d')

weather_all = []
for region, info in REGION_COORDS.items():
    print(f"Fetching weather: {region} ({info['city']})...")
    w = fetch_weather(info['lat'], info['lon'], start_str, end_str)
    if w is not None:
        w['Region'] = region
        weather_all.append(w)

weather_df = pd.concat(weather_all, ignore_index=True)
df = df.merge(
    weather_df[['Date', 'Region', 'temp_mean', 'temp_max', 'temp_min',
                'precipitation', 'wind_speed', 'sunshine']],
    on=['Date', 'Region'], how='left'
)
print("After weather merge:", df.shape)
print("Missing values:", df[['temp_mean', 'precipitation']].isna().sum().to_dict())

# Real Russian public holidays 2022-2024
RUSSIAN_HOLIDAYS = [
    # 2022
    '2022-01-01','2022-01-02','2022-01-03','2022-01-04','2022-01-05',
    '2022-01-06','2022-01-07','2022-01-08','2022-02-23','2022-03-07',
    '2022-03-08','2022-05-02','2022-05-03','2022-05-09','2022-05-10',
    '2022-06-13','2022-11-04','2022-12-31',
    # 2023
    '2023-01-01','2023-01-02','2023-01-03','2023-01-04','2023-01-05',
    '2023-01-06','2023-01-07','2023-01-08','2023-01-09','2023-02-23',
    '2023-02-24','2023-03-08','2023-05-01','2023-05-08','2023-05-09',
    '2023-06-12','2023-11-06','2023-12-31',
    # 2024
    '2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05',
    '2024-01-06','2024-01-07','2024-01-08','2024-02-23','2024-03-08',
    '2024-04-29','2024-04-30','2024-05-01','2024-05-09','2024-05-10',
    '2024-06-12','2024-11-04','2024-12-30','2024-12-31',
]

holidays_set = set(pd.to_datetime(RUSSIAN_HOLIDAYS))
pre_holidays  = {pd.Timestamp(d) - pd.Timedelta(days=1) for d in holidays_set}
post_holidays = {pd.Timestamp(d) + pd.Timedelta(days=1) for d in holidays_set}

df['real_holiday']  = df['Date'].isin(holidays_set).astype(int)
df['pre_holiday']   = df['Date'].isin(pre_holidays).astype(int)
df['post_holiday']  = df['Date'].isin(post_holidays).astype(int)
df['holiday_week']  = df['Date'].apply(
    lambda d: int(any(abs((d - h).days) <= 3 for h in holidays_set))
)

# Calendar features
df['day_of_week']    = df['Date'].dt.dayofweek
df['month']          = df['Date'].dt.month
df['quarter']        = df['Date'].dt.quarter
df['week_of_year']   = df['Date'].dt.isocalendar().week.astype(int)
df['is_weekend']     = (df['Date'].dt.dayofweek >= 5).astype(int)
df['day_of_month']   = df['Date'].dt.day
df['is_month_start'] = (df['Date'].dt.day <= 5).astype(int)
df['is_month_end']   = (df['Date'].dt.day >= 25).astype(int)

# Weather interaction features
df['is_cold']      = (df['temp_mean'] < 0).astype(int)
df['heavy_rain']   = (df['precipitation'] > 10).astype(int)
df['cold_holiday'] = df['is_cold'] * df['real_holiday']

df.to_csv('data_enriched.csv', index=False)
print(f"\ndata_enriched.csv saved: {df.shape}")
print(f"New features: weather (6) + holidays (4) + calendar (8) + interactions (3)")

