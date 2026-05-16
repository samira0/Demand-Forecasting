#!/usr/bin/env python
# coding: utf-8

# # ENSEMBLE-MODELS

# In[5]:


# !pip install catboost


# In[6]:


import time
import warnings
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


# In[7]:


random_state = 42
train_path = "train.csv"
test_path = "test.csv"
raw_data_path = "data_enriched_with_features.csv"
target = "Units Sold"
n_cv_splits = 5
save_results = True
results_dir = "results"


# In[8]:


warnings.filterwarnings('ignore')


# In[9]:


df_train = pd.read_csv(train_path, index_col="Date")
df_train.index = pd.to_datetime(df_train.index)


# In[10]:


df_test = pd.read_csv(test_path, index_col="Date")
df_test.index = pd.to_datetime(df_test.index)


# In[11]:


y_train = df_train[target]
X_train = df_train.drop(target, axis=1)
y_test = df_test[target]
X_test = df_test.drop(target, axis=1)


# In[12]:


def weighted_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


# In[13]:


def create_pred_list(model_class, X_train, y_train, X_test, y_test, **kwargs):
    y_pred = pd.Series(index=y_test.index, dtype=float)
    for i in range(len(y_test)):
        X_tr = pd.concat([X_train, X_test.iloc[:i]], axis=0)
        y_tr = pd.concat([y_train, y_test.iloc[:i]], axis=0)
        X_next = X_test.iloc[i : i + 1]
        model = model_class(**kwargs)
        model.fit(X_tr, y_tr)
        y_pred.iloc[i] = model.predict(X_next)[0]
    return y_pred


# In[14]:


def tune_and_evaluate(
    model_class,
    param_grid,
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    cv_splits=None,
    do_expanding=True,
    do_batch=True,
):
    """
    Run GridSearchCV, then evaluate with expanding-window and/or batch predict.
    Returns (best_params, batch_predictions_series or None).
    """
    cv_splits = cv_splits or n_cv_splits
    t0 = time.time()
    grid = GridSearchCV(
        estimator=model_class(),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_root_mean_squared_error",
        verbose=0,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[{name}] Best params: {grid.best_params_}  ({elapsed:.1f}s)")

    best_params = grid.best_params_

    if do_expanding:
        print(f"\n--- {name} (expanding window, one-step-ahead) ---")
        y_pred_exp = create_pred_list(
            model_class, X_train, y_train, X_test, y_test, **best_params
        )
        show_metrics(y_test, y_pred_exp, title=f"{name} — expanding", plot=True)

    y_pred_batch = None
    if do_batch:
        print(f"\n--- {name} (batch on full test) ---")
        model = model_class(**best_params)
        model.fit(X_train, y_train)
        y_pred_batch = pd.Series(model.predict(X_test), index=y_test.index)
        show_metrics(y_test, y_pred_batch, title=f"{name} — batch", plot=True)

    return best_params, y_pred_batch


# In[15]:


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.where(denominator > 0, numerator / denominator, 0.0)
    return float(np.mean(out))


# In[16]:


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "SMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "WMAPE": weighted_mean_absolute_percentage_error(y_true, y_pred),
    }


# In[17]:


def show_metrics(y_true, y_pred, title=None, plot=True):
    metrics = compute_metrics(y_true, y_pred)
    if title:
        print(title)
    print(f"MAE:   {metrics['MAE']:.3f}")
    print(f"RMSE:  {metrics['RMSE']:.3f}")
    print(f"SMAPE: {metrics['SMAPE']:.3f}")
    print(f"WMAPE: {metrics['WMAPE']:.3f}")

    if plot:
        plt.figure(figsize=(10, 4))
        plt.title(title or "True vs predicted")
        plt.plot(y_true, label="true")
        plt.plot(y_pred, label="pred")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


# ## ML-Models

# In[18]:


MODEL_CONFIGS = [
    (
        "RandomForest",
        RandomForestRegressor,
        {
            "n_estimators": [100, 200],
            "random_state": [random_state],
            "n_jobs": [-1],
        },
    ),
    (
        "GradientBoosting",
        GradientBoostingRegressor,
        {
            "n_estimators": [100, 200],
            "random_state": [random_state],
        },
    ),
    (
        "LightGBM",
        LGBMRegressor,
        {
            "n_estimators": [100, 200],
            "random_state": [random_state],
            "verbosity": [-1],
        },
    ),
    (
        "XGBoost",
        XGBRegressor,
        {
            "n_estimators": [100, 200],
            "random_state": [random_state],
            "n_jobs": [-1],
        },
    ),
    (
        "CatBoost",
        CatBoostRegressor,
        {
            "n_estimators": [100, 200],
            "random_state": [random_state],
            "logging_level": ["Silent"],
        },
    ),
]


# In[19]:


def run_all_models(do_expanding=True, do_batch=True):
    """Tune and evaluate each model; return list of (name, batch_predictions)."""
    results = []
    for name, model_class, param_grid in MODEL_CONFIGS:
        _, y_pred_batch = tune_and_evaluate(
            model_class,
            param_grid,
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            do_expanding=do_expanding,
            do_batch=do_batch,
        )
        results.append((name, y_pred_batch))
    return results


# In[20]:


if save_results:
    os.makedirs(results_dir, exist_ok=True)

model_predictions = run_all_models(do_expanding=True, do_batch=True)

batch_preds = {name: pred for name, pred in model_predictions 
               if pred is not None}

if batch_preds and save_results:
    rows = []
    for name, pred in batch_preds.items():
        rows.append({"Model": name, **compute_metrics(y_test, pred)})
    pd.DataFrame(rows).sort_values("MAE").to_csv(
        os.path.join(results_dir, "ensemble_batch_results.csv"), 
        index=False)

ensemble_pred = pd.concat(
    batch_preds.values(), axis=1).mean(axis=1)
ensemble_pred = pd.Series(ensemble_pred, index=y_test.index)
show_metrics(y_test, ensemble_pred, 
             title="Ensemble (mean) — batch", plot=True)

estimators = [
    ("rf",   RandomForestRegressor(
        random_state=random_state, n_estimators=200, n_jobs=-1)),
    ("gbr",  GradientBoostingRegressor(
        random_state=random_state, n_estimators=200)),
    ("lgbm", LGBMRegressor(
        random_state=random_state, n_estimators=200, verbosity=-1)),
    ("xgb",  XGBRegressor(
        random_state=random_state, n_estimators=200, n_jobs=-1)),
    ("cat",  CatBoostRegressor(
        random_state=random_state, n_estimators=200, 
        logging_level="Silent")),
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0, random_state=random_state),
    n_jobs=-1,
    passthrough=False,
)
stack.fit(X_train, y_train)
y_pred_stack = pd.Series(stack.predict(X_test), index=y_test.index)
show_metrics(y_test, y_pred_stack, title="Stacking — batch", plot=True)

if save_results:
    pd.DataFrame({
        "y_true": y_test.values, 
        "y_pred": y_pred_stack.values
    }, index=y_test.index).to_csv(
        os.path.join(results_dir, "stacking_predictions.csv"))
    
raw = pd.read_csv('retail_store_inventory.csv')
raw["Date"] = pd.to_datetime(raw["Date"])
rep = raw[
    (raw["Store ID"] == "S001") & 
    (raw["Product ID"] == "P0001")
].copy().set_index("Date").sort_index()

if "Demand Forecast" in rep.columns:
    existing      = rep["Demand Forecast"].reindex(y_test.index).dropna()
    aligned_true  = y_test.reindex(existing.index)

    best_name     = min(
        batch_preds.keys(),
        key=lambda n: compute_metrics(y_test, batch_preds[n])["MAE"]
    )
    best_single   = batch_preds[best_name].reindex(existing.index)
    mean_ens      = ensemble_pred.reindex(existing.index)
    stack_aligned = y_pred_stack.reindex(existing.index)

    rows = [
        {"Model": "Existing Demand Forecast",
         **compute_metrics(aligned_true, existing)},
        {"Model": f"Best single ({best_name})",
         **compute_metrics(aligned_true, best_single)},
        {"Model": "Mean ensemble",
         **compute_metrics(aligned_true, mean_ens)},
        {"Model": "Stacking",
         **compute_metrics(aligned_true, stack_aligned)},
    ]

    comp = pd.DataFrame(rows).sort_values("MAE")
    print("Comparison with existing Demand Forecast (S001/P0001)")
    print(f"Aligned rows: {len(existing)} / {len(y_test)}")
    print(comp.round(3).to_string(index=False))

    if save_results:
        comp.to_csv(
            os.path.join(results_dir, 
                         "comparison_existing_forecast.csv"), 
            index=False)
else:
    print("Demand Forecast column not found in raw data.")


# In[21]:


if __name__ == "__main__":
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    model_predictions = run_all_models(do_expanding=True, do_batch=True)

    # Meta-ensemble: average of batch predictions from all models
    batch_preds = {name: pred for name, pred in model_predictions if pred is not None}
    if batch_preds and save_results:
        rows = []
        for name, pred in batch_preds.items():
            rows.append({"Model": name, **compute_metrics(y_test, pred)})
        pd.DataFrame(rows).sort_values("MAE").to_csv(
            os.path.join(results_dir, "ensemble_batch_results.csv"), index=False
        )

    if batch_preds:
        ensemble_pred = pd.concat(batch_preds.values(), axis=1).mean(axis=1)
        ensemble_pred = pd.Series(ensemble_pred, index=y_test.index)
        show_metrics(y_test, ensemble_pred, title="Ensemble (mean) — batch", plot=True)
        if save_results:
            pd.DataFrame(
                {"y_true": y_test.values, "y_pred": ensemble_pred.values},
                index=y_test.index,
            ).to_csv(os.path.join(results_dir, "ensemble_mean_predictions.csv"))

        # Stacking ensemble (Ridge meta-learner)
        estimators = [
            ("rf", RandomForestRegressor(random_state=random_state, n_estimators=200, n_jobs=-1)),
            ("gbr", GradientBoostingRegressor(random_state=random_state, n_estimators=200)),
            ("lgbm", LGBMRegressor(random_state=random_state, n_estimators=200, verbosity=-1)),
            ("xgb", XGBRegressor(random_state=random_state, n_estimators=200, n_jobs=-1)),
            ("cat", CatBoostRegressor(random_state=random_state, n_estimators=200, logging_level="Silent")),
        ]
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0, random_state=random_state),
            n_jobs=-1,
            passthrough=False,
        )
        stack.fit(X_train, y_train)
        y_pred_stack = pd.Series(stack.predict(X_test), index=y_test.index)
        show_metrics(y_test, y_pred_stack, title="Stacking — batch", plot=True)
        if save_results:
            pd.DataFrame(
                {"y_true": y_test.values, "y_pred": y_pred_stack.values},
                index=y_test.index,
            ).to_csv(os.path.join(results_dir, "stacking_predictions.csv"))

    # Comparison with existing-system forecast (Demand Forecast) for S001–P0001 aligned by date
    if batch_preds and os.path.exists(raw_data_path) and not y_test.empty:
        raw = pd.read_csv(raw_data_path)
        raw["Date"] = pd.to_datetime(raw["Date"])
        rep = raw[(raw["Store ID"] == "S001") & (raw["Product ID"] == "P0001")].copy()
        rep = rep.set_index("Date").sort_index()
        if "Demand Forecast" in rep.columns:
            existing = rep["Demand Forecast"].reindex(y_test.index).dropna()
            aligned_true = y_test.reindex(existing.index)

            # best single model (batch) by MAE
            best_name = min(batch_preds.keys(), key=lambda n: compute_metrics(y_test, batch_preds[n])["MAE"])
            best_single = batch_preds[best_name].reindex(existing.index)
            mean_ens = ensemble_pred.reindex(existing.index)
            rows = [
                {"Model": "Existing Demand Forecast", **compute_metrics(aligned_true, existing)},
                {"Model": f"Best single ({best_name})", **compute_metrics(aligned_true, best_single)},
                {"Model": "Mean ensemble", **compute_metrics(aligned_true, mean_ens)},
            ]
            if "y_pred_stack" in locals():
                rows.append(
                    {
                        "Model": "Stacking",
                        **compute_metrics(aligned_true, y_pred_stack.reindex(existing.index)),
                    }
                )

            comp = pd.DataFrame(rows).sort_values("MAE")
            print("\nComparison with existing Demand Forecast (S001–P0001)\n")
            print(f"Aligned rows: {len(existing)} / {len(y_test)}")
            print(comp.round(3).to_string(index=False))
            if save_results:
                comp.to_csv(os.path.join(results_dir, "comparison_existing_forecast.csv"), index=False)


# ## Proper Hyperparameter Tuning For Individual Models Comparison

# #### Uses TimeSeriesSplit + RandomizedSearchCV for every model

# In[22]:


import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import scipy.stats as stats
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv', index_col='Date')
df_train.index = pd.to_datetime(df_train.index)
df_test = pd.read_csv('test.csv', index_col='Date')
df_test.index = pd.to_datetime(df_test.index)

target = 'Units Sold'
y_train = df_train[target]
X_train = df_train.drop(target, axis=1)
y_test  = df_test[target]
X_test  = df_test.drop(target, axis=1)

# TimeSeriesSplit for tuning
tscv = TimeSeriesSplit(n_splits=3)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Metric helpers
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom))

def show_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))
    sm   = smape(np.array(y_true), np.array(y_pred))
    wm   = wmape(np.array(y_true), np.array(y_pred))
    print(f"{name:30s}  MAE={mae:.3f}  RMSE={rmse:.3f}"
          f"  SMAPE={sm:.3f}  WMAPE={wm:.3f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse,
            'SMAPE': sm, 'WMAPE': wm}

# Hyperparameter search spaces
param_grids = {

    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators':      [100, 200, 300],
            'max_depth':         [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
            'max_features':      ['sqrt', 'log2', 0.5],
        }
    },

    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample':     [0.7, 0.8, 1.0],
            'min_samples_leaf': [1, 2, 5],
        }
    },

    'LightGBM': {
        'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves':    [15, 31, 63],
            'subsample':     [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        }
    },

    'XGBoost': {
        'model': XGBRegressor(random_state=42, n_jobs=-1,
                              verbosity=0, eval_metric='mae'),
        'params': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample':     [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
        }
    },

    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, verbose=0),
        'params': {
            'iterations':    [100, 200, 300],
            'depth':         [3, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg':   [1, 3, 5, 7],
            'subsample':     [0.7, 0.8, 1.0],
        }
    },
}

#  Tune and evaluate each model
print("HYPERPARAMETER TUNING WITH TimeSeriesSplit (n=3)")

all_results = []
best_models = {}

for name, config in param_grids.items():
    print(f"\nTuning {name}...")

    search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=30,              # 30 random combinations per model
        cv=tscv,
        scoring=mae_scorer,
        n_jobs=-1,
        random_state=42,
        refit=True,             # refit on full training data
    )
    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV MAE: {-search.best_score_:.3f}")

    # Evaluate on test set
    y_pred = search.best_estimator_.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    r = show_metrics(name, y_test, y_pred)
    r['best_params'] = str(search.best_params_)
    r['cv_mae'] = -search.best_score_
    all_results.append(r)
    best_models[name] = search.best_estimator_

# Summary
print("FINAL COMPARISON — ALL MODELS AFTER TUNING")

results_df = pd.DataFrame(all_results).sort_values('MAE')
print(results_df[['model', 'MAE', 'RMSE', 'SMAPE',
                   'WMAPE', 'cv_mae']].to_string(index=False))

results_df.to_csv('tuned_model_results.csv', index=False)
print("\nThe winner after fair tuning:", results_df.iloc[0]['model'])


# ## Error Analysis and Diagnostics

# In[23]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

target_col = 'Units Sold'
if 'Date' in train.columns:
    train = train.drop(columns=['Date'])
    test = test.drop(columns=['Date'])

y_train = train[target_col]
y_test = test[target_col]
X_train = train.drop(columns=[target_col]).select_dtypes(include=[np.number])
X_test = test.drop(columns=[target_col]).select_dtypes(include=[np.number])

common = list(set(X_train.columns) & set(X_test.columns))
X_train, X_test = X_train[common], X_test[common]

RANDOM_STATE = 42


# In[24]:


# Stacking Ensemble
estimators = [
    ("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1)),
    ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200)),
    ("lgbm", LGBMRegressor(random_state=RANDOM_STATE, n_estimators=200, verbosity=-1)),
    ("xgb", XGBRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1)),
    ("cat", CatBoostRegressor(random_state=RANDOM_STATE, n_estimators=200, logging_level="Silent")),
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
    n_jobs=-1,
    passthrough=False,
)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

# Stacking Residuals
res_stack = y_test.values - y_pred_stack

print("\n STACKING RESIDUALS")
print(f"Mean:              {np.mean(res_stack):.4f}")
print(f"Std:               {np.std(res_stack):.4f}")
print(f"Min / Max:         {np.min(res_stack):.2f} / {np.max(res_stack):.2f}")

lb = acorr_ljungbox(res_stack, lags=10, return_df=True)
print(f"Ljung-Box p:       {lb['lb_pvalue'].iloc[-1]:.4f}")

sh_stat, sh_p = stats.shapiro(res_stack)
print(f"Shapiro-Wilk p:    {sh_p:.4f}")

med_pred = np.median(y_pred_stack)
print(f"Res std (low pred):  {np.std(res_stack[y_pred_stack <= med_pred]):.4f}")
print(f"Res std (high pred): {np.std(res_stack[y_pred_stack > med_pred]):.4f}")

# Base Learner Performance
print("\n BASE LEARNER PERFORMANCE")
base_maes = {}
for name, model in estimators:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    base_maes[name] = mae
    print(f"{name:6s}: MAE = {mae:.3f}")

# Meta-learner Coefficients
ridge = stack.final_estimator_

print("\n META-LEARNER (Ridge) COEFFICIENTS")
for name, coef in zip(['rf', 'gbr', 'lgbm', 'xgb', 'cat'], ridge.coef_):
    print(f"{name:6s}: {coef:.4f}")
print(f"Intercept: {ridge.intercept_:.4f}")

# Feature Importance from GradientBoosting
gb = GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200)
gb.fit(X_train, y_train)

imp = pd.Series(gb.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print("\n FEATURE IMPORTANCE (GradientBoosting)")
print("Top 10 individual features:")
print(imp.head(10).round(4))

# Grouped
groups = {
    'Units Ordered': [c for c in imp.index if 'units_ordered' in c.lower() or 'units ordered' in c.lower()],
    'Price': [c for c in imp.index if 'price' in c.lower()],
    'Discount': [c for c in imp.index if 'discount' in c.lower()],
    'Calendar': [c for c in imp.index if any(x in c.lower() for x in ['day', 'month', 'season', 'week'])],
    'Holiday/Promotion': [c for c in imp.index if any(x in c.lower() for x in ['holiday', 'promotion'])],
    'External': [c for c in imp.index if any(x in c.lower() for x in ['weather', 'region', 'competitor', 'category'])],
}
groups['Other'] = [c for c in imp.index if not any(c in g for g in groups.values())]

print("\nGrouped importance:")
for gname, cols in groups.items():
    if cols:
        gsum = imp[cols].sum()
        print(f"  {gname:20s}: {gsum:.4f} ({gsum/imp.sum()*100:.1f}%)")

# Prediction Horizon
print("\n PREDICTION HORIZON SENSITIVITY")
gb_pred = gb.predict(X_test)
print(f"Horizon  1: MAE = {mean_absolute_error(y_test, gb_pred):.2f}")

for h in [7, 14, 30]:
    idx = np.arange(0, len(y_test), h)
    if len(idx) > 3:
        mae_h = mean_absolute_error(y_test.iloc[idx], gb_pred[idx])
        print(f"Horizon {h:2d}: MAE = {mae_h:.2f} (n={len(idx)} points)")
    else:
        print(f"Horizon {h:2d}: (insufficient data)")


# In[25]:


fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# Top panel: time series of actual vs predicted
axes[0].plot(y_test.values, label='Actual', color='steelblue', linewidth=1)
axes[0].plot(y_pred_stack, label='Stacking Ensemble', 
             color='tomato', linewidth=1, linestyle='--')
axes[0].set_title(
    'Actual vs Predicted: Stacking Ensemble (Test Set)', fontsize=13
)
axes[0].set_xlabel('Time index')
axes[0].set_ylabel('Units Sold')
axes[0].legend()

# Bottom panel: residuals
residuals = y_test.values - y_pred_stack
axes[1].plot(residuals, color='grey', linewidth=0.8)
axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
axes[1].set_title('Residuals (Actual − Predicted)', fontsize=13)
axes[1].set_xlabel('Time index')
axes[1].set_ylabel('Residual')

plt.tight_layout()
plt.show()

