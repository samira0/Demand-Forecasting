#!/usr/bin/env python
# coding: utf-8

# # Multi-Series Modeling 

# In[7]:


import pandas as pd
import numpy as np
import warnings
from itertools import product as iterproduct
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')


# In[8]:


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom))

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    sm   = smape(y_true, y_pred)
    wm   = wmape(y_true, y_pred)
    return {'model': label, 'MAE': mae, 'RMSE': rmse,
            'SMAPE': sm, 'WMAPE': wm}


# In[9]:


df = pd.read_csv('data_enriched_with_features.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(
    ['Store ID', 'Product ID', 'Date']
).reset_index(drop=True)

selected = pd.read_csv('selected_series.csv')
series_list = list(zip(selected['store'], selected['product']))

print(f"Loaded: {df.shape}")
print(f"Modeling {len(series_list)} series")
print("Series:", series_list)


# In[10]:


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

# Exogenous columns for ARIMAX — subset of FEATURE_COLS
# all available before the forecast is made
EXOG_COLS = [
    'Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion',
    'temp_mean', 'precipitation', 'wind_speed',
    'real_holiday', 'pre_holiday', 'post_holiday',
    'day_of_week', 'month', 'is_weekend', 'is_cold',
]

TARGET = 'Units Sold'
tscv   = TimeSeriesSplit(n_splits=3)

# Check all feature columns are present
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"WARNING — missing columns: {missing}")
else:
    print(f"All {len(FEATURE_COLS)} feature columns present.")


# ## ARIMA, ARIMAX and GB Per Series

# In[11]:


def fit_best_arima(train_y, max_p=2, max_q=2):
    """
    Grid search over ARIMA orders, select by AIC.
    Reduced search space (p,q up to 2) for computational efficiency.
    Justified by full-panel analysis: 88% of series have no
    autocorrelation structure, so high-order models are unnecessary.
    """
    best_aic   = np.inf
    best_order = (0, 0, 0)
    for p, d, q in iterproduct(
            range(max_p + 1), [0, 1], range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            r = ARIMA(train_y, order=(p, d, q)).fit(
                method_kwargs={'warn_convergence': False})
            if r.aic < best_aic:
                best_aic   = r.aic
                best_order = (p, d, q)
        except:
            continue
    try:
        r0 = ARIMA(train_y, order=(0, 0, 0)).fit()
        if r0.aic < best_aic:
            best_aic   = r0.aic
            best_order = (0, 0, 0)
    except:
        pass
    return best_order, best_aic


all_results = []

for store, product in series_list:
    print(f"\n  {store} / {product}")

    mask = (df['Store ID'] == store) & (df['Product ID'] == product)
    s_df = df[mask].copy().sort_values('Date').reset_index(drop=True)

    prof = selected.loc[
        (selected['store'] == store) &
        (selected['product'] == product), 'profile'
    ].values[0]
    print(f"  Profile: {prof}")

    s_df = s_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split    = int(len(s_df) * 0.8)
    train_df = s_df.iloc[:split]
    test_df  = s_df.iloc[split:]

    y_train_arr = train_df['Units Sold'].values
    y_test_arr  = test_df['Units Sold'].values

    print(f"  Train: {len(y_train_arr)} days | Test: {len(y_test_arr)} days")

    # Naive baseline
    naive_pred = np.full(len(y_test_arr), y_train_arr.mean())
    r = metrics(y_test_arr, naive_pred, 'Naive_Mean')
    r.update({'store': store, 'product': product, 'profile': prof})
    all_results.append(r)
    print(f"  Naive MAE={r['MAE']:.2f}  WMAPE={r['WMAPE']:.3f}")

    # ARIMA — BATCH
    # Batch prediction consistent with ensemble evaluation protocol
    best_order, best_aic = fit_best_arima(y_train_arr)
    print(f"  Best ARIMA order: {best_order}  (AIC={best_aic:.1f})")
    try:
        arima_model = ARIMA(
            y_train_arr,
            order=best_order
        ).fit(method_kwargs={'warn_convergence': False})
        arima_preds = np.clip(
            arima_model.forecast(steps=len(y_test_arr)), 0, None)
        r = metrics(y_test_arr, arima_preds, f'ARIMA{best_order}')
        r.update({'store': store, 'product': product, 'profile': prof})
        all_results.append(r)
        print(f"  ARIMA{best_order}: MAE={r['MAE']:.2f}  "
              f"WMAPE={r['WMAPE']:.3f}")
    except Exception as e:
        print(f"  ARIMA failed: {e}")

    # ARIMAX — BATCH
    exog_train = train_df[EXOG_COLS].values
    exog_test  = test_df[EXOG_COLS].values
    try:
        arimax_model = ARIMA(
            y_train_arr,
            order=best_order,
            exog=exog_train
        ).fit(method_kwargs={'warn_convergence': False})
        arimax_preds = np.clip(
            arimax_model.forecast(
                steps=len(y_test_arr),
                exog=exog_test), 0, None)
        r = metrics(y_test_arr, arimax_preds, f'ARIMAX{best_order}')
        r.update({'store': store, 'product': product, 'profile': prof})
        all_results.append(r)
        print(f"  ARIMAX{best_order}: MAE={r['MAE']:.2f}  "
              f"WMAPE={r['WMAPE']:.3f}")
    except Exception as e:
        print(f"  ARIMAX failed: {e}")

    # GradientBoosting
    X_tr = train_df[FEATURE_COLS]
    y_tr = train_df['Units Sold'].values
    X_te = test_df[FEATURE_COLS]
    y_te = test_df['Units Sold'].values

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42)
    gb.fit(X_tr, y_tr)
    gb_preds = np.clip(gb.predict(X_te), 0, None)

    r = metrics(y_te, gb_preds, 'GradientBoosting')
    r.update({'store': store, 'product': product, 'profile': prof})
    all_results.append(r)
    print(f"  GradientBoosting: MAE={r['MAE']:.2f}  "
          f"WMAPE={r['WMAPE']:.3f}")


# In[12]:


results_df = pd.DataFrame(all_results).round(3)

print("\nRESULTS SUMMARY BY SERIES AND PROFILE")
summary = results_df.pivot_table(
    index=['store', 'product', 'profile'],
    columns='model', values='MAE'
).reset_index()
print(summary.to_string(index=False))

print("\nAVERAGE MAE BY PROFILE AND MODEL")
profile_avg = results_df.groupby(
    ['profile', 'model'])['MAE'].mean().unstack()
print(profile_avg.round(3).to_string())

print("\nKEY FINDINGS")
for profile in results_df['profile'].unique():
    p_df   = results_df[results_df['profile'] == profile]
    naive  = p_df[p_df['model'] == 'Naive_Mean']['MAE'].mean()
    arima  = p_df[p_df['model'].str.startswith('ARIMA(')]['MAE'].mean()
    arimax = p_df[p_df['model'].str.startswith('ARIMAX')]['MAE'].mean()
    gb     = p_df[p_df['model'] == 'GradientBoosting']['MAE'].mean()
    print(f"\nProfile: {profile}")
    print(f"  Naive:  {naive:.2f}")
    if not np.isnan(arima):
        print(f"  ARIMA:  {arima:.2f}  "
              f"({(naive - arima)/naive*100:+.1f}% vs naive)")
    if not np.isnan(arimax):
        print(f"  ARIMAX: {arimax:.2f}  "
              f"({(naive - arimax)/naive*100:+.1f}% vs naive)")
    if not np.isnan(gb):
        print(f"  GB:     {gb:.2f}  "
              f"({(naive - gb)/naive*100:+.1f}% vs naive)")

results_df.to_csv('multi_series_arima_gb_results.csv', index=False)


# ## Tuned GB vs Tuned Stacking (10 Series)

# In[13]:


TUNED_PARAMS = {
    'GradientBoosting': dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ),
    'RandomForest': dict(
        n_estimators=300, max_depth=None, min_samples_split=5,
        min_samples_leaf=1, max_features=0.5,
        random_state=42, n_jobs=-1
    ),
    'LightGBM': dict(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        num_leaves=31, subsample=0.7, colsample_bytree=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    ),
    'XGBoost': dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.7, colsample_bytree=1.0, min_child_weight=1,
        random_state=42, n_jobs=-1, verbosity=0
    ),
    'CatBoost': dict(
        iterations=200, depth=3, learning_rate=0.05,
        l2_leaf_reg=5, subsample=0.7,
        random_state=42, verbose=0
    ),
}

def make_models():
    return {
        'GradientBoosting': GradientBoostingRegressor(
            **TUNED_PARAMS['GradientBoosting']),
        'RandomForest':     RandomForestRegressor(
            **TUNED_PARAMS['RandomForest']),
        'LightGBM':         LGBMRegressor(
            **TUNED_PARAMS['LightGBM']),
        'XGBoost':          XGBRegressor(
            **TUNED_PARAMS['XGBoost']),
        'CatBoost':         CatBoostRegressor(
            **TUNED_PARAMS['CatBoost']),
    }


# In[14]:


def fit_stacking(X_tr, y_tr, X_te, models):
    """
    Temporal stacking using manual TimeSeriesSplit OOF loop.
    Avoids cross_val_predict which does not support TimeSeriesSplit.
    Meta-learner is Ridge fitted on out-of-fold predictions.
    """
    n_models   = len(models)
    oof_preds  = np.zeros((len(X_tr), n_models))
    test_preds = np.zeros((len(X_te), n_models))
    splits     = list(tscv.split(X_tr))

    for i, (name, model) in enumerate(models.items()):
        fold_oof = np.zeros(len(X_tr))
        for train_idx, val_idx in splits:
            m = type(model)(**model.get_params())
            m.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
            fold_oof[val_idx] = np.clip(
                m.predict(X_tr.iloc[val_idx]), 0, None)
        oof_preds[:, i] = fold_oof
        model.fit(X_tr, y_tr)
        test_preds[:, i] = np.clip(model.predict(X_te), 0, None)

    # Only use rows filled by at least one fold
    filled = np.any(oof_preds != 0, axis=1)
    meta   = Ridge(alpha=1.0)
    meta.fit(oof_preds[filled], y_tr.iloc[filled])
    stack_pred = np.clip(meta.predict(test_preds), 0, None)

    return stack_pred, test_preds, meta.coef_


# In[15]:


print("TUNED GB vs TUNED STACKING — 10 STRATIFIED SERIES")

records    = []
train_list = []
test_list  = []
selected_keys = set(zip(selected['store'], selected['product']))

for (store, product), grp in df.groupby(['Store ID', 'Product ID']):
    if (store, product) not in selected_keys:
        continue

    profile = selected.loc[
        (selected['store'] == store) &
        (selected['product'] == product), 'profile'
    ].values[0]

    grp_feat = grp.dropna(subset=FEATURE_COLS).copy()
    grp_feat = grp_feat.sort_values('Date').reset_index(drop=True)

    missing = [c for c in FEATURE_COLS if c not in grp_feat.columns]
    if missing:
        print(f"  WARNING {store}/{product} missing: {missing}")
        continue

    cutoff = int(len(grp_feat) * 0.8)
    train  = grp_feat.iloc[:cutoff].copy()
    test   = grp_feat.iloc[cutoff:].copy()

    X_tr = train[FEATURE_COLS]
    y_tr = train[TARGET]
    X_te = test[FEATURE_COLS]
    y_te = test[TARGET]

    naive_mae = mean_absolute_error(
        y_te, np.full(len(y_te), y_tr.mean()))

    print(f"\n  {store}/{product}  [{profile}]")
    print(f"    Train: {len(train)} | Test: {len(test)}")
    print(f"    Naive MAE: {naive_mae:.3f}")

    record = {
        'store': store, 'product': product,
        'profile': profile, 'naive_mae': naive_mae,
    }

    # Tuned GradientBoosting
    gb = GradientBoostingRegressor(**TUNED_PARAMS['GradientBoosting'])
    gb.fit(X_tr, y_tr)
    gb_pred = np.clip(gb.predict(X_te), 0, None)
    record['gb_mae']   = mean_absolute_error(y_te, gb_pred)
    record['gb_wmape'] = wmape(y_te.values, gb_pred)
    print(f"    GB (tuned):       MAE={record['gb_mae']:.3f}  "
          f"WMAPE={record['gb_wmape']:.3f}")

    # Tuned Stacking
    models_inst = make_models()
    stack_pred, _, meta_coefs = fit_stacking(
        X_tr, y_tr, X_te, models_inst)
    record['stack_mae']   = mean_absolute_error(y_te, stack_pred)
    record['stack_wmape'] = wmape(y_te.values, stack_pred)
    print(f"    Stacking (tuned): MAE={record['stack_mae']:.3f}  "
          f"WMAPE={record['stack_wmape']:.3f}")

    winner = ('Stacking' if record['stack_mae'] < record['gb_mae']
              else 'GradientBoosting')
    impr   = ((record['gb_mae'] - record['stack_mae']) /
               record['gb_mae'] * 100)
    print(f"    Winner: {winner}  Stacking vs GB: {impr:+.1f}%")

    record['winner'] = winner
    records.append(record)

    train['_store']   = test['_store']   = store
    train['_product'] = test['_product'] = product
    train['_profile'] = test['_profile'] = profile
    train_list.append(train)
    test_list.append(test)

stack_results_df = pd.DataFrame(records)


# In[16]:


print("SUMMARY — PER-SERIES GB vs STACKING")

print("\nBy series:")
print(stack_results_df[[
    'store', 'product', 'profile',
    'naive_mae', 'gb_mae', 'stack_mae', 'winner'
]].to_string(index=False))

print("\nAverage MAE by profile:")
print(stack_results_df.groupby('profile')[
    ['naive_mae', 'gb_mae', 'stack_mae']
].mean().round(3).to_string())

print("\nOverall averages:")
naive_avg = stack_results_df['naive_mae'].mean()
gb_avg    = stack_results_df['gb_mae'].mean()
stack_avg = stack_results_df['stack_mae'].mean()
print(f"  Naive:    {naive_avg:.3f}")
print(f"  GB:       {gb_avg:.3f}  "
      f"({(naive_avg - gb_avg)/naive_avg*100:.1f}% vs naive)")
print(f"  Stacking: {stack_avg:.3f}  "
      f"({(naive_avg - stack_avg)/naive_avg*100:.1f}% vs naive)")
print(f"\nStacking wins: "
      f"{(stack_results_df['winner'] == 'Stacking').sum()} "
      f"/ {len(stack_results_df)} series")


# ## Panel Model - GB vs Stacking

# In[17]:


print("PANEL MODEL — TRAINED JOINTLY ON 10 SERIES")

panel_train = pd.concat(train_list, ignore_index=True)
panel_test  = pd.concat(test_list,  ignore_index=True)
print(f"Panel train: {len(panel_train)} | test: {len(panel_test)}")

X_pan_tr = panel_train[FEATURE_COLS]
y_pan_tr = panel_train[TARGET]
X_pan_te = panel_test[FEATURE_COLS]
y_pan_te = panel_test[TARGET]

# Panel GB
gb_panel = GradientBoostingRegressor(**TUNED_PARAMS['GradientBoosting'])
gb_panel.fit(X_pan_tr, y_pan_tr)
pan_gb_pred = np.clip(gb_panel.predict(X_pan_te), 0, None)
pan_gb_mae  = mean_absolute_error(y_pan_te, pan_gb_pred)
pan_gb_wm   = wmape(y_pan_te.values, pan_gb_pred)
print(f"\nPanel GB:       MAE={pan_gb_mae:.3f}  WMAPE={pan_gb_wm:.3f}")

# Panel Stacking
pan_models    = make_models()
pan_stack, _, pan_coefs = fit_stacking(
    X_pan_tr, y_pan_tr, X_pan_te, pan_models)
pan_stack_mae = mean_absolute_error(y_pan_te, pan_stack)
pan_stack_wm  = wmape(y_pan_te.values, pan_stack)
print(f"Panel Stacking: MAE={pan_stack_mae:.3f}  "
      f"WMAPE={pan_stack_wm:.3f}")

print("\nPanel meta-learner coefficients:")
for name, coef in zip(TUNED_PARAMS.keys(), pan_coefs):
    print(f"  {name:<20} {coef:.3f}")

panel_winner = ('Stacking' if pan_stack_mae < pan_gb_mae
                else 'GradientBoosting')
print(f"\nPanel winner: {panel_winner}")


# In[18]:


stack_results_df.to_csv('stacking_vs_gb_results.csv', index=False)
results_df.to_csv('multi_series_arima_gb_results.csv', index=False)


# In[19]:


from scipy import stats

print("STATISTICAL SIGNIFICANCE — STACKING vs GRADIENTBOOSTING")

gb_maes    = stack_results_df['gb_mae'].values
stack_maes = stack_results_df['stack_mae'].values
differences = gb_maes - stack_maes  # positive = GB worse = stacking better

print(f"\nPer-series MAE values:")
for i, row in stack_results_df.iterrows():
    diff = row['gb_mae'] - row['stack_mae']
    winner = 'Stacking' if diff > 0 else 'GB'
    print(f"  {row['store']}/{row['product']}: "
          f"GB={row['gb_mae']:.3f}  "
          f"Stack={row['stack_mae']:.3f}  "
          f"diff={diff:+.3f}  ({winner})")

print(f"\nOverall:")
print(f"  GB mean MAE:       {gb_maes.mean():.3f}")
print(f"  Stacking mean MAE: {stack_maes.mean():.3f}")
print(f"  Mean diff (GB-Stack): {differences.mean():+.3f}")
print(f"  Stacking wins: "
      f"{(differences > 0).sum()} / {len(differences)} series")
print(f"  GB wins:       "
      f"{(differences < 0).sum()} / {len(differences)} series")

# Paired t-test
t_stat, t_p = stats.ttest_rel(gb_maes, stack_maes)
print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value:     {t_p:.4f}")

# Wilcoxon signed-rank test
# alternative='greater' tests whether GB > Stacking i.e. stacking is better
w_stat, w_p_two = stats.wilcoxon(gb_maes, stack_maes,
                                  alternative='two-sided')
w_stat, w_p_stacking = stats.wilcoxon(gb_maes, stack_maes,
                                       alternative='greater')
w_stat, w_p_gb       = stats.wilcoxon(gb_maes, stack_maes,
                                       alternative='less')

print(f"\nWilcoxon signed-rank test:")
print(f"  Two-sided p:              {w_p_two:.4f}")
print(f"  Stacking better (p):      {w_p_stacking:.4f}")
print(f"  GB better (p):            {w_p_gb:.4f}")

print(f"\nConclusion:")
if w_p_stacking < 0.05:
    print("  Stacking significantly outperforms GB (p < 0.05)")
elif w_p_gb < 0.05:
    print("  GB significantly outperforms Stacking (p < 0.05)")
elif w_p_two < 0.05:
    print("  Significant difference detected but direction unclear")
else:
    print("  No statistically significant difference between "
          "Stacking and GB")

