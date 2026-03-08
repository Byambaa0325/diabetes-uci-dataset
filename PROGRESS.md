# Coursework #1 Progress

## Critical Issue — Target Variable

**The current target encoding is wrong for Tasks 2–4.**

The coursework asks: classify *no readmission* vs *any readmission* (combining `<30` and `>30`).
`preprocessor.py` currently does:
```python
df[col] = (df[col] == "<30").astype(int)   # ← WRONG for tasks 2–4
```
This only marks `<30` as positive and treats `>30` as negative.

Correct encoding for tasks 2–4:
```python
df[col] = (df[col] != "NO").astype(int)    # <30 or >30 → 1, NO → 0
```

Class balance changes significantly:
| Encoding | Positive rate |
|---|---|
| `<30` only (current) | ~11% |
| `<30` or `>30` (correct) | ~46% |

**Task 5 is likely where `<30`-only prediction is introduced** as the "alternative" pipeline. So the current encoding is probably correct for Task 5 but wrong for Tasks 2–4.

---

## Task 1 — Dataset Description (15%) ✅ Done

**Notebook:** `notebooks/01_eda.ipynb`

| Sub-task | Status | Notes |
|---|---|---|
| 1.1 Dataset characteristics | ✅ | 101,766 rows × 50 cols, data types, distributions, demographics, clinical features, medication rates, correlation heatmap |
| 1.2 Challenges | ✅ | Missing data (`weight` 97%, `A1Cresult` 83%, `max_glu_serum` 95%), class imbalance (~11% `<30`), 30% repeat patient encounters, overlapping feature distributions |

**Needs for report:** Extract the key figures and statistics from the notebook for the 3-page report.

---

## Task 2 — Data Assembling and Pre-processing (10%)

| Sub-task | Status | Notes |
|---|---|---|
| 2.1 Assemble X, y | ✅ | `src/data/loader.py`, `preprocessor.py`, `features.py` |
| 2.2 Cleaning and pre-processing | ✅ | Drops (`weight`, `payer_code`, near-zero meds), deduplication on `patient_nbr`, ordinal encoding for age, one-hot for race/discharge/admission, scaler fitted on train only |
| 2.3 Strategy for diag_1/2/3 | ✅ | ICD-9 grouped into 8 clinical categories (Circulatory, Respiratory, Diabetes, etc.) using Strack et al. (2014) CCS-inspired grouping in `features.py:_encode_diagnoses` |
| **Target encoding** | ❌ | Must fix to `(df[col] != "NO").astype(int)` for tasks 2–4 |

---

## Task 3 — Machine Learning Pipeline (40%)

| Sub-task | Status | Notes |
|---|---|---|
| 3.1 Metrics choice + justification | ⚠️ Partial | ROC-AUC, F1, Recall (macro), Recall(<30), Precision implemented in `src/evaluation/metrics.py`. Justification text needed for report. |
| 3.2 Linear SVM baseline + 2 models | ❌ | **RBF SVM** is registered, not **Linear SVM** as required. Need to add `SVC(kernel='linear')` or `LinearSVC`. Two additional models: Random Forest + one more (LR or GBM). |
| 3.3 Pipeline with hyperparameter optimisation + CV | ❌ | **Cross-validation is not implemented.** Current pipeline does a single 80/20 holdout split. Task requires k-fold CV with multiple performance estimates per hyperparameter value. |
| 3.4 Performance vs hyperparameter plots | ⚠️ Partial | `exp_04_hyperparameter_sweep.ipynb` plots exist for LR (C), RF (max_depth), MLP (architecture). But these are single-split, not CV. Need to redo with CV mean ± std. |
| 3.5 Mean CV performance table (with std) | ❌ | Requires CV implementation first. |
| 3.6 Results description | ❌ | Needs to be written once CV results are available. |

### What's built for Task 3 (reusable)
- `src/pipeline/` — full 4-stage pipeline (load → featurize → train → evaluate)
- `sweep()` — shares featurization across hyperparameter configs, runs all in sequence
- `src/models/registry.py` — model registry (LR, RF, GBM, SVM-RBF, MLP)
- `src/models/trainers/` — sklearn + torch trainers with W&B logging
- `src/evaluation/bias_variance.py` — bootstrap bias-variance decomposition

### Gaps to fill
1. **Add linear SVM** to registry (`SVC(kernel='linear')` or `LinearSVC`)
2. **Implement k-fold CV** — either wrap `sweep()` or use sklearn's `cross_validate` directly
3. **Re-run sweeps** under CV to get mean ± std per hyperparameter value

---

## Task 4 — Model Interpretation (10%) ❌ Not started

| Sub-task | Status | Notes |
|---|---|---|
| 4.1 Feature importance plots | ❌ | Need: LR coefficients, RF feature_importances_, permutation importance or SHAP for SVM |
| 4.2 Similarities/differences across models | ❌ | Report text |
| 4.3 Clinical validity discussion | ❌ | Report text |

---

## Task 5 — Alternative Pipeline (25%) ❌ Not started

Task description cut off in the brief — need to confirm full wording. Based on context, likely:
- Use best model from Task 3
- Switch target to `<30` only (early readmission) — this is where the current target encoding fits
- The `recall_lt30` metric already tracks this

**Likely structure:**
- Keep the pipeline as-is (target = `<30` only)
- Add cross-validation
- Compare against Task 3 results — discuss the harder subproblem

---

## Codebase Status

### What works
| Component | File(s) |
|---|---|
| Data loading | `src/data/loader.py` |
| Cleaning + deduplication | `src/data/preprocessor.py` |
| Full feature engineering (94 features) | `src/data/features.py` |
| Basic featurizer (18 features) | `src/data/features_basic.py` |
| Pipeline stages | `src/pipeline/load.py`, `featurize.py`, `run_train.py`, `run_evaluate.py` |
| Pipeline runner + sweep | `src/pipeline/runner.py` |
| Sklearn trainer | `src/models/trainers/sklearn_trainer.py` |
| PyTorch MLP trainer + early stopping | `src/models/trainers/torch_trainer.py` |
| Evaluation metrics | `src/evaluation/metrics.py` |
| Bias-variance decomposition | `src/evaluation/bias_variance.py` |
| W&B logging | `src/utils/logger.py` |

### Known issues
| Issue | Location | Fix needed |
|---|---|---|
| Target encoding wrong for tasks 2–4 | `preprocessor.py:26` | `!= "NO"` instead of `== "<30"` |
| No cross-validation | Entire pipeline | Add k-fold CV wrapper |
| Linear SVM missing | `src/models/registry.py` | Add `LinearSVC` or `SVC(kernel='linear')` entry |
| No feature importance output | `src/evaluation/` | New module or notebook |
| Unshuffled train/val split | `pipeline/load.py:20` | Add `random_state`-seeded shuffle before split |
| Stale `src/pipeline.py` on disk | `src/` | Delete to prevent import confusion |
| `02_bias_variance.ipynb` uses old 41-feature build | `notebooks/02_bias_variance.ipynb` | Re-run with current 94-feature featurizer |

---

## Immediate Next Steps (priority order)

1. **Fix target encoding** in `preprocessor.py` — affects all downstream results
2. **Add linear SVM** to the model registry
3. **Implement cross-validation** — core requirement for Task 3.3, 3.4, 3.5
4. **Feature importance** — Task 4.1
5. Re-run bias-variance notebook with correct target + current featurizer
6. Write report sections once above are done
