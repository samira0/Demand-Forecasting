#!/usr/bin/env python
# coding: utf-8

# # Business Impact Analysis

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


# In[21]:


results = pd.read_csv('multi_series_arima_gb_results.csv')
analysis  = pd.read_csv('series_analysis.csv')
selected  = pd.read_csv('selected_series.csv')
df        = pd.read_csv('data_enriched.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Results shape:  {results.shape}")
print(f"Analysis shape: {analysis.shape}")


# In[22]:


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# ### Demand Patterns By Category And Region

# In[23]:


# Basic demand stats by category
cat_stats = df.groupby('Category')['Units Sold'].agg(
    ['mean', 'std', 'median', 'count']
).round(2)
cat_stats['cv'] = (cat_stats['std'] / cat_stats['mean'] * 100).round(1)
cat_stats.columns = ['Mean Sales', 'Std', 'Median', 'Count', 'CV%']
cat_stats = cat_stats.sort_values('Mean Sales', ascending=False)
print("\nDemand by Category:")
print(cat_stats.to_string())

# Basic demand stats by region
reg_stats = df.groupby('Region')['Units Sold'].agg(
    ['mean', 'std', 'median']
).round(2)
reg_stats['cv'] = (reg_stats['std'] / reg_stats['mean'] * 100).round(1)
reg_stats.columns = ['Mean Sales', 'Std', 'Median', 'CV%']
reg_stats = reg_stats.sort_values('Mean Sales', ascending=False)
print("\nDemand by Region:")
print(reg_stats.to_string())

# Demand by category and region combined
cat_reg = df.groupby(['Category', 'Region'])['Units Sold'].mean().round(2)
print("\nDemand by Category × Region:")
print(cat_reg.unstack().round(2).to_string())

# Plot: demand by category
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cat_stats['Mean Sales'].plot(
    kind='bar', ax=axes[0], color='#4C72B0',
    edgecolor='white', linewidth=1.2
)
axes[0].set_title('Mean Daily Sales by Category', fontsize=12)
axes[0].set_ylabel('Units Sold')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=30)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[0].set_axisbelow(True)

reg_stats['Mean Sales'].plot(
    kind='bar', ax=axes[1], color='#2CA02C',
    edgecolor='white', linewidth=1.2
)
axes[1].set_title('Mean Daily Sales by Region', fontsize=12)
axes[1].set_ylabel('Units Sold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=30)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig('demand_by_category_region.png', dpi=180, bbox_inches='tight')
plt.show()


# ### Holiday And Weather Impact On Demand

# In[24]:


# Holiday impact
if 'real_holiday' in df.columns:
    holiday_impact = df.groupby('real_holiday')['Units Sold'].agg(
        ['mean', 'std', 'median', 'count']
    ).round(2)
    holiday_impact.index = ['Regular day', 'Public holiday']
    holiday_impact.columns = ['Mean Sales', 'Std', 'Median', 'Count']
    print("\nHoliday vs Regular day demand:")
    print(holiday_impact.to_string())

    pct_change = (holiday_impact.loc['Public holiday', 'Mean Sales'] /
                  holiday_impact.loc['Regular day', 'Mean Sales'] - 1) * 100
    print(f"\nHoliday effect: {pct_change:+.1f}% vs regular days")

    # Pre and post holiday
    pre_impact  = df.groupby('pre_holiday')['Units Sold'].mean()
    post_impact = df.groupby('post_holiday')['Units Sold'].mean()
    base = df[df['real_holiday'] == 0]['Units Sold'].mean()
    print(f"\nPre-holiday demand:  {pre_impact.get(1, base):.1f} "
          f"({(pre_impact.get(1, base)/base - 1)*100:+.1f}% vs baseline)")
    print(f"Post-holiday demand: {post_impact.get(1, base):.1f} "
          f"({(post_impact.get(1, base)/base - 1)*100:+.1f}% vs baseline)")

# Weather impact
if 'temp_mean' in df.columns:
    print("\nWeather impact on demand:")

    # Temperature bins
    df['temp_bin'] = pd.cut(
        df['temp_mean'],
        bins=[-40, -10, 0, 10, 20, 40],
        labels=['Very cold\n(<-10°C)', 'Cold\n(-10 to 0°C)',
                'Cool\n(0-10°C)', 'Warm\n(10-20°C)', 'Hot\n(>20°C)']
    )
    temp_impact = df.groupby('temp_bin')['Units Sold'].agg(
        ['mean', 'count']
    ).round(2)
    temp_impact.columns = ['Mean Sales', 'Count']
    print("\nSales by temperature range:")
    print(temp_impact.to_string())

    # Precipitation impact
    rain_impact = df.groupby('heavy_rain')['Units Sold'].mean().round(2)
    rain_impact.index = ['No heavy rain', 'Heavy rain (>10mm)']
    print("\nHeavy rain effect on sales:")
    print(rain_impact.to_string())
    if 1 in rain_impact.index.map(lambda x: 'Heavy rain (>10mm)' == x):
        pass
    rain_vals = df.groupby('heavy_rain')['Units Sold'].mean()
    if len(rain_vals) == 2:
        rain_effect = (rain_vals[1] / rain_vals[0] - 1) * 100
        print(f"Heavy rain effect: {rain_effect:+.1f}% vs normal days")

    # Cold day impact
    cold_impact = df.groupby('is_cold')['Units Sold'].mean().round(2)
    if len(cold_impact) == 2:
        cold_effect = (cold_impact[1] / cold_impact[0] - 1) * 100
        print(f"\nCold day (<0°C) effect: {cold_effect:+.1f}% vs warm days")

# Plot: weather and holiday impact
if 'temp_mean' in df.columns and 'real_holiday' in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Temperature
    temp_means = df.groupby('temp_bin')['Units Sold'].mean()
    temp_means.plot(kind='bar', ax=axes[0], color='#4C72B0',
                    edgecolor='white', linewidth=1.2)
    axes[0].set_title('Sales by Temperature', fontsize=11)
    axes[0].set_ylabel('Mean Units Sold')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
    axes[0].set_axisbelow(True)

    # Holiday comparison
    h_data = df.groupby('real_holiday')['Units Sold'].mean()
    bars = axes[1].bar(['Regular day', 'Public holiday'],
                       h_data.values,
                       color=['#4C72B0', '#E05C5C'],
                       edgecolor='white', linewidth=1.2)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{bar.get_height():.1f}',
                     ha='center', va='bottom', fontsize=11)
    axes[1].set_title('Holiday vs Regular Day', fontsize=11)
    axes[1].set_ylabel('Mean Units Sold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
    axes[1].set_axisbelow(True)

    # Pre/during/post holiday
    df['holiday_period'] = 'Regular'
    df.loc[df['pre_holiday'] == 1, 'holiday_period'] = 'Pre-holiday'
    df.loc[df['real_holiday'] == 1, 'holiday_period'] = 'Holiday'
    df.loc[df['post_holiday'] == 1, 'holiday_period'] = 'Post-holiday'
    order = ['Pre-holiday', 'Holiday', 'Post-holiday', 'Regular']
    period_data = df.groupby('holiday_period')['Units Sold'].mean()
    period_data = period_data.reindex(
        [o for o in order if o in period_data.index]
    )
    colors = ['#F39C12', '#E05C5C', '#8496B0', '#4C72B0']
    bars = axes[2].bar(period_data.index, period_data.values,
                       color=colors[:len(period_data)],
                       edgecolor='white', linewidth=1.2)
    for bar in bars:
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{bar.get_height():.1f}',
                     ha='center', va='bottom', fontsize=10)
    axes[2].set_title('Demand Around Holidays', fontsize=11)
    axes[2].set_ylabel('Mean Units Sold')
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
    axes[2].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('weather_holiday_impact.png', dpi=180, bbox_inches='tight')
    plt.show()


# ### Series Profile Distribution Across Categories

# In[25]:


print(analysis.columns.tolist())
# Merge analysis with category/region info
cat_region_info = df.groupby(
    ['Store ID', 'Product ID']
)[['Category', 'Region']].first().reset_index()
cat_region_info.columns = ['store', 'product', 'category', 'region']

analysis_enriched = analysis.merge(cat_region_info,
                                    on=['store', 'product'], how='left')

print("\nProfile distribution by Category:")
profile_cat = pd.crosstab(
    analysis['category'],
    analysis['profile']
)
print(profile_cat.to_string())

print("\nProfile distribution by Region:")
profile_reg = pd.crosstab(
    analysis['region'],
    analysis['profile']
)
print(profile_reg.to_string())

print("\nARIMA useful by Category:")
arima_cat = analysis.groupby('category')['arima_useful'].agg(
    ['sum', 'count', 'mean']
).round(3)
arima_cat.columns = ['ARIMA useful', 'Total', 'Proportion']
print(arima_cat.to_string())


# ### Safety Stock Calculation

# In[28]:


# Safety stock comparison — meaningful model configurations only
naive_mae         = 86.02   # multi-series naive baseline
arimax_mae        = 16.251  # single series ARIMAX with exogenous features
gb_panel_mae      = 32.70   # panel GB — realistic production estimate (10 series)
gb_single_mae     = 2.51    # single series tuned GB — best achievable

models_ss = {
    'Naive baseline':              naive_mae,
    'ARIMAX (single series)':      arimax_mae,
    'GB panel (production est.)':  gb_panel_mae,
    'GB single series (tuned)':    gb_single_mae,
}

print(f"\nDataset statistics:")
print(f"  Mean daily demand per series: {mean_daily_demand:.1f} units")
print(f"  Std daily demand per series:  {std_daily_demand:.1f} units")
print(f"  Number of store-product series: {n_series}")
print(f"  Service level target: 95% (Z = {Z})")
print(f"  Replenishment lead time: {lead_time} days")

print(f"\nSafety stock per series (units):")
ss_values = {}
for name, mae in models_ss.items():
    ss = safety_stock(mae)
    ss_values[name] = ss
    print(f"  {name:<35} {ss:.1f} units/series")

print(f"\nTotal safety stock across all {n_series} series:")
for name, ss in ss_values.items():
    print(f"  {name:<35} {ss * n_series:,.0f} units")

print(f"\nSafety stock reduction vs Naive:")
naive_ss = ss_values['Naive baseline']
for name, ss in ss_values.items():
    if name != 'Naive baseline':
        red = (naive_ss - ss) / naive_ss * 100
        print(f"  {name:<35} {red:.1f}% reduction")

print(f"\nEstimated annual holding cost:")
for name, ss in ss_values.items():
    cost = ss * n_series * daily_holding * 365
    saving = (ss_values['Naive baseline'] * n_series * daily_holding * 365) - cost
    print(f"  {name:<35} {cost:,.0f} m.u./year  "
          f"(saves {saving:,.0f} vs naive)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

models_labels = [
    'Naive\nbaseline',
    'ARIMAX\n(single)',
    'GB panel\n(production)',
    'GB single\n(tuned)'
]
ss_total   = [46798, 13813, 17790, 1366]
cost_total = [129010, 38079, 49042, 3764]
colors     = ['#8496B0', '#4C72B0', '#3A80C8', '#2CA02C']

# Safety stock
bars = axes[0].bar(models_labels, ss_total, color=colors,
                   edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, ss_total):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 200,
                 f'{val:,.0f}',
                 ha='center', va='bottom', fontsize=9)
axes[0].set_title('Total Safety Stock Across 100 Series', fontsize=11)
axes[0].set_ylabel('Units')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[0].set_axisbelow(True)

# Annual holding cost
bars2 = axes[1].bar(models_labels, cost_total, color=colors,
                    edgecolor='white', linewidth=1.2)
for bar, val in zip(bars2, cost_total):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 500,
                 f'{val:,.0f}',
                 ha='center', va='bottom', fontsize=9)
axes[1].set_title('Annual Holding Cost Across 100 Series', fontsize=11)
axes[1].set_ylabel('Monetary units / year')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig('safety_stock_analysis.png', dpi=180, bbox_inches='tight')
plt.show()


# ### Seasonal And Temporal Demand Patterns

# In[27]:


# Monthly patterns
monthly = df.groupby('month')['Units Sold'].mean().round(2)
month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
monthly.index = [month_names[m] for m in monthly.index]
print("\nMean sales by month:")
print(monthly.to_string())
print(f"  Peak month:    {monthly.idxmax()} ({monthly.max():.1f} units)")
print(f"  Trough month:  {monthly.idxmin()} ({monthly.min():.1f} units)")
print(f"  Seasonal range:{monthly.max() - monthly.min():.1f} units "
      f"({(monthly.max()/monthly.min()-1)*100:.1f}%)")

# Day of week patterns
dow = df.groupby('day_of_week')['Units Sold'].mean().round(2)
day_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu',
             4:'Fri', 5:'Sat', 6:'Sun'}
dow.index = [day_names[d] for d in dow.index]
print("\nMean sales by day of week:")
print(dow.to_string())
print(f"  Best day:  {dow.idxmax()} ({dow.max():.1f} units)")
print(f"  Worst day: {dow.idxmin()} ({dow.min():.1f} units)")

# Quarter patterns
quarterly = df.groupby('quarter')['Units Sold'].mean().round(2)
quarterly.index = [f'Q{q}' for q in quarterly.index]
print("\nMean sales by quarter:")
print(quarterly.to_string())

# Plot: temporal patterns
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

monthly.plot(kind='bar', ax=axes[0], color='#4C72B0',
             edgecolor='white', linewidth=1.2)
axes[0].set_title('Mean Sales by Month', fontsize=11)
axes[0].set_ylabel('Mean Units Sold')
axes[0].tick_params(axis='x', rotation=30)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[0].set_axisbelow(True)

dow.plot(kind='bar', ax=axes[1], color='#4C72B0',
         edgecolor='white', linewidth=1.2)
axes[1].set_title('Mean Sales by Day of Week', fontsize=11)
axes[1].set_ylabel('Mean Units Sold')
axes[1].tick_params(axis='x', rotation=0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[1].set_axisbelow(True)

quarterly.plot(kind='bar', ax=axes[2], color='#4C72B0',
               edgecolor='white', linewidth=1.2)
axes[2].set_title('Mean Sales by Quarter', fontsize=11)
axes[2].set_ylabel('Mean Units Sold')
axes[2].tick_params(axis='x', rotation=0)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].yaxis.grid(True, color='#EBEBEB', linewidth=0.8)
axes[2].set_axisbelow(True)

plt.tight_layout()
plt.savefig('temporal_patterns.png', dpi=180, bbox_inches='tight')
plt.show()

