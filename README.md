# Applied AI Coursework

Diabetes readmission prediction using the UCI Diabetes 130-US Hospitals dataset.

## Dataset

[UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — 100k+ inpatient encounters. Target: readmission within 30 days.

---

# Pipeline Config Guide

Everything is driven by a single config dict. Here's how to use it.

---

## The minimum you need

```python
from src.pipeline import run_pipeline

result = run_pipeline({
    "model": "logistic_regression",
})
```

That's it. One key. Everything else has a sensible default. It will load the data, engineer features, train, evaluate on a val set, and hand you back a `PipelineResult`.

---

## Models you can use

```python
"model": "logistic_regression"   # sklearn LogisticRegression
"model": "random_forest"         # sklearn RandomForestClassifier
"model": "gradient_boosting"     # sklearn GradientBoostingClassifier
"model": "linear_svm"            # sklearn LinearSVC (fast, no probability calibration)
"model": "svm"                   # sklearn SVC (RBF kernel, needs subsample on large data)
"model": "mlp"                   # PyTorch MLP (needs a few extra keys — see below)
"model": "tabicl"                # TabICL tabular foundation model (needs pip install tabicl)
```

Sklearn docs for each:
- [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

### TabICL — tabular foundation model

TabICL is an in-context learning model for tabular data. It needs a separate install:

```bash
pip install tabicl
```

Once installed it's available as any other model — no extra config needed:

```python
result = run_pipeline({
    "model": "tabicl",
    "name":  "tabicl_baseline",
})
```

If `tabicl` isn't installed the registry just skips it silently — the rest of the pipeline is unaffected. Trying to use `"model": "tabicl"` without it installed will raise a clear `ValueError` from the registry lookup.

---

## Passing hyperparameters to sklearn models

Anything in `model_params` gets forwarded directly to the sklearn constructor:

```python
{
    "model": "logistic_regression",
    "model_params": {
        "C": 0.1,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
}
```

Works exactly the same for `random_forest`, `gradient_boosting`, `linear_svm`, `svm`.

---

## MLP config

The MLP has its own keys (no `model_params`):

```python
{
    "model":       "mlp",
    "hidden_dims": [256, 128, 64],   # layer sizes — add/remove as you like
    "dropout":     0.3,              # applied after every hidden layer
    "lr":          1e-3,
    "batch_size":  256,
    "epochs":      100,
    "patience":    10,               # early stopping — set 0 to disable
}
```

`input_dim` is filled in automatically from the feature count, you don't need to set it.

---

## Choosing your featurizer

```python
"featurizer": "full"    # default — ICD grouping, medication direction, engineered features (~94)
"featurizer": "basic"   # minimal — raw numerics + simple ordinal/binary only (~18)
```

Leave it out and you get `"full"`.

---

## Splitting the data

```python
"split_ratio": 0.8     # 80% train, 20% val — default
```

If you want a proper held-out test set:

```python
"split_ratio": 0.8,    # of the non-test portion
"test_ratio":  0.15,   # 15% carved out first, never touched until stage_test
```

With `test_ratio` set, `result.test_metrics` will be populated alongside `result.metrics` (val).

---

## Cross-validation

Add one key:

```python
{
    "model":     "logistic_regression",
    "cv_splits": 5,
}
```

`result.metrics` will contain mean values (`roc_auc`, `f1`, etc.) and `result.metrics['roc_auc_std']` for the spread. Raw per-fold dicts are in `result.fold_metrics`.

Without `cv_splits` you get a single holdout split — faster, good for iteration.

---

## Large datasets — subsampling

SVM and other slow models struggle on 57k rows. Subsample the training set:

```python
{
    "model":     "svm",
    "subsample": 15_000,    # stratified random sample of X_train only
    "model_params": {"kernel": "rbf", "probability": True, "class_weight": "balanced"},
}
```

Val (and test if set) are never subsampled — they always reflect the full holdout.

---

## Evaluation options

```python
"threshold": 0.3    # fixed decision threshold
"threshold": None   # default — automatically picks the F1-optimal threshold
"plot":      True   # render ROC, PR curve, confusion matrix (good in notebooks)
```

---

## W&B logging

```python
"name":          "my_experiment",          # run name in W&B and on disk
"wandb_project": "applied-ai-coursework",  # W&B project
```

Leave `wandb_project` out and it defaults to `"applied-ai-coursework"`.

---

## Reading the result

```python
result = run_pipeline(config)

result.metrics          # {'roc_auc': 0.71, 'f1': 0.43, 'recall': 0.68, 'recall_lt30': 0.55, ...}
result.test_metrics     # same shape, on test set — None if no test_ratio
result.run_dir          # Path to saved model, config.json, metrics.json
result.model            # fitted model object
result.feature_bundle   # FeatureBundle with X_train, X_val, X_test, n_features, scaler
result.elapsed_s        # total wall time
result.is_cv            # True if cv_splits was set
result.fold_metrics     # list of per-fold dicts — None for single-split runs
```

---

## Common patterns

### Quick baseline
```python
result = run_pipeline({"model": "logistic_regression"})
```

### Hyperparameter sweep (fast, single split)
```python
from src.pipeline import sweep

results = sweep(
    base_config={"model": "logistic_regression", "featurizer": "full"},
    param_grid=[{"name": f"lr_C{c}", "model_params": {"C": c}} for c in [0.01, 0.1, 1.0, 10.0]],
)
# features built once, 4 models trained
```

### Final evaluation with CV + test set
```python
result = run_pipeline({
    "model":        "logistic_regression",
    "model_params": {"C": 1.0, "class_weight": "balanced"},
    "cv_splits":    5,
    "test_ratio":   0.15,
    "name":         "lr_final",
})
print(result.metrics)       # CV means on val folds
print(result.test_metrics)  # held-out test — look at this last
```

### Featurizer ablation
```python
from src.pipeline import sweep

results = sweep(
    base_config={"model": "random_forest"},
    param_grid=[
        {"featurizer": "full",  "name": "rf_full"},
        {"featurizer": "basic", "name": "rf_basic"},
    ],
)
# featurize called once per featurizer, not once per model
```

### CV sweep over hyperparameters (for Task 3.4 / 3.5)
```python
from src.pipeline import cross_validate_sweep

param_grid = [{"name": f"lr_C{c}", "model_params": {"C": c}} for c in [0.01, 0.1, 1.0, 10.0]]

results = cross_validate_sweep(
    base_config={"model": "logistic_regression", "featurizer": "full"},
    param_grid=param_grid,
    n_splits=5,
)

for cfg, r in zip(param_grid, results):
    s = r["summary"]
    print(f"C={cfg['model_params']['C']}  "
          f"ROC-AUC={s['roc_auc_mean']:.3f}±{s['roc_auc_std']:.3f}")
```

---

## All config keys at a glance

**Core**
- `model` *(str, required)* — which model to train; see model list above. `"tabicl"` requires `pip install tabicl`
- `model_params` *(dict, default `{}`)* — passed straight to the sklearn constructor

**Data**
- `split_ratio` *(float, default `0.8`)* — train fraction of the non-test data
- `test_ratio` *(float, default `0.0`)* — held-out test fraction carved out before everything else; `0` means no test set
- `subsample` *(int, default `None`)* — cap on training rows, stratified; val and test are never touched

**Features**
- `featurizer` *(str, default `"full"`)* — `"full"` for full feature engineering (~94 features) or `"basic"` for minimal encoding (~18 features)

**Evaluation**
- `cv_splits` *(int, default `None`)* — number of CV folds; `None` means a single holdout split
- `threshold` *(float|None, default `None`)* — decision threshold; `None` picks the F1-optimal threshold automatically
- `plot` *(bool, default `False`)* — render ROC, PR curve and confusion matrix inline

**Logging**
- `name` *(str, default: model key)* — run name used in W&B and the run directory on disk
- `wandb_project` *(str, default `"applied-ai-coursework"`)* — W&B project to log to

**MLP only**
- `hidden_dims` *(list[int], default `[64, 32]`)* — hidden layer sizes, e.g. `[256, 128, 64]`
- `dropout` *(float, default `0.3`)* — dropout probability applied after each hidden layer
- `lr` *(float, required for MLP)* — Adam learning rate
- `epochs` *(int, required for MLP)* — maximum training epochs
- `batch_size` *(int, required for MLP)* — mini-batch size
- `patience` *(int, default `10`)* — early stopping patience in epochs; `0` disables it
