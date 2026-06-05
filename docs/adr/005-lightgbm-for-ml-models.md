# ADR-005: LightGBM for Price and Occupancy Models

**Date:** 2026-06-05
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

Phase 3 requires regression models for:
1. **Price prediction** — nightly rate (EUR), right-skewed, range ~€20–€500
2. **Occupancy prediction** — annual occupancy rate ∈ [0, 1]

The feature matrix has ~28 numeric features + room type dummies. Dataset size: ~15K listings after joining `listings` with `listing_features` and filtering for non-null targets.

Requirements: fast training on a laptop, good handling of sparse/missing features, interpretable feature importances (needed for Phase 4 SHAP), pickle-serializable.

Options evaluated:

| Option | Pros | Cons |
|--------|------|------|
| **LightGBM** | Fast, handles nulls natively, SHAP-compatible, small install, early stopping | Hyperparameter-sensitive, less interpretable than linear models |
| CatBoost | Handles categoricals natively, often SOTA | Slower training, larger install, categorical encoding less relevant here (few cats) |
| XGBoost | Mature, SHAP-compatible | Slower than LightGBM on this scale, similar accuracy |
| Scikit-learn GradientBoosting | No extra deps | Much slower, no GPU, no early stopping |
| Linear Regression | Interpretable, fast | Underfits on geospatial features with non-linear interactions |
| Random Forest | Robust, low-tuning | Higher memory, slower inference, SHAP slower |

---

## Decision

Use **LightGBM** (`LGBMRegressor`) for both price and occupancy models.

- Price: log-transform target (`np.log(price)`) before training; exponentiate predictions back. Reduces right-skew, stabilizes RMSE.
- Occupancy: no transform; clip predictions to [0, 1] post-inference.
- Both: `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63`, early stopping at 50 rounds on 20% validation split.
- Models serialized with `pickle` to `data/models/{price,occupancy}_model.pkl`.

---

## Consequences

**Positive:**
- Training completes in seconds on a laptop for this dataset size
- Early stopping prevents overfitting without manual epoch tuning
- Native missing value handling — no imputation required in training (features fill null → 0 only for `prepare_X_y_*`)
- SHAP TreeSHAP works out-of-the-box with LightGBM models (Phase 4 ready)
- Feature importances available via `model.feature_importances_`

**Negative:**
- Pickle format is not forward-compatible across major LightGBM versions. Mitigation: pin `lightgbm>=4.3.0` in `pyproject.toml`; re-train when upgrading.
- `num_leaves=63` may overfit on small neighbourhoods with few listings. Mitigation: monitor validation RMSE/MAE; tune if Phase 4 analysis shows per-neighbourhood degradation.

---

## References

- [LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- [LightGBM + SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/LightGBM%20tutorial.html)
- `src/geoai/models/price.py`, `src/geoai/models/occupancy.py`
