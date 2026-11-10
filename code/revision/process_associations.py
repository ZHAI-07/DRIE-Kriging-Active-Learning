"""Descriptive process associations for fixed six-structure response means.

No causal identification is claimed. Full-data standardization is used only
for descriptive coefficients. All held-out predictions and coefficient ranges
use input/response standardization fitted within each training fold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"outputs/revision/evidence"
SOURCE = ROOT/"data/raw/user_confirmed_table_2026-11-09.json"
FEATURES = ["Coil", "Platen", "SF6", "passivation_pressure"]
SCENARIOS = {"all24": [], "without24": [24], "without24_and2": [24, 2], "without24_and13": [24, 13]}


def fit_standardized(X, y, model_name):
    xs = StandardScaler().fit(X)
    ys = StandardScaler().fit(y[:, None])
    model = LinearRegression() if model_name == "OLS" else Ridge(alpha=1.)
    model.fit(xs.transform(X), ys.transform(y[:, None]).ravel())
    return xs, ys, model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    array = np.asarray(raw["rows"], float)
    all_ids = array[:, 0].astype(int)
    response_names = ["mean_CD", "mean_AR"]
    # Fixed arithmetic means in original physical units: no PCA fit or
    # data-dependent response weights are used in this analysis.
    all_Y = np.column_stack([array[:, 5::2].mean(axis=1), array[:, 6::2].mean(axis=1)])
    full_coefficients, fold_coefficients, predictions, design_rows = [], [], [], []
    for scenario, exclusions in SCENARIOS.items():
        keep = ~np.isin(all_ids, exclusions)
        ids, X, Y = all_ids[keep], array[keep, 1:5], all_Y[keep]
        groups = np.array([2 if i in [2, 13] else i for i in ids])
        Z = StandardScaler().fit_transform(X)
        H = np.column_stack([np.ones(len(X)), Z])
        corr = pd.DataFrame(X, columns=FEATURES).corr()
        corr.to_csv(OUT/f"process_input_correlation_{scenario}.csv", encoding="utf-8-sig")
        lower = corr.values[np.tril_indices(4, -1)]
        design_rows.append(dict(scenario=scenario, n_records=len(ids), n_source_groups=len(np.unique(groups)),
                                centered_input_rank=int(np.linalg.matrix_rank(Z)),
                                design_rank_with_intercept=int(np.linalg.matrix_rank(H)),
                                standardized_input_condition_number=float(np.linalg.cond(Z)),
                                standardized_design_condition_number=float(np.linalg.cond(H)),
                                raw_input_condition_number=float(np.linalg.cond(X)),
                                max_absolute_input_correlation=float(np.abs(lower).max())))
        for response_index, response in enumerate(response_names):
            y = Y[:, response_index]
            for model_name in ["OLS", "Ridge_alpha1"]:
                _, _, model = fit_standardized(X, y, model_name)
                for feature, beta in zip(FEATURES, model.coef_):
                    full_coefficients.append(dict(scenario=scenario, response=response, model=model_name,
                                                  feature=feature, standardized_coefficient_full=float(beta)))
                for group in np.unique(groups):
                    test = groups == group
                    train = ~test
                    xs, ys, fitted = fit_standardized(X[train], y[train], model_name)
                    pred = ys.inverse_transform(fitted.predict(xs.transform(X[test]))[:, None]).ravel()
                    for feature, beta in zip(FEATURES, fitted.coef_):
                        fold_coefficients.append(dict(scenario=scenario, response=response, model=model_name,
                                                      heldout_source_group=int(group), n_train=int(train.sum()),
                                                      feature=feature, standardized_coefficient_training_fold=float(beta)))
                    for index, estimate in zip(np.flatnonzero(test), pred):
                        predictions.append(dict(scenario=scenario, response=response, model=model_name,
                                                experiment_id=int(ids[index]), heldout_source_group=int(group),
                                                y_true=float(y[index]), y_pred=float(estimate),
                                                mean_baseline_prediction=float(y[train].mean())))
    fold = pd.DataFrame(fold_coefficients)
    fold.to_csv(OUT/"process_coefficients_training_folds.csv", index=False, encoding="utf-8-sig")
    coefficient_summary = fold.groupby(["scenario", "response", "model", "feature"]).agg(
        n_training_folds=("standardized_coefficient_training_fold", "count"),
        training_fold_min=("standardized_coefficient_training_fold", "min"),
        training_fold_median=("standardized_coefficient_training_fold", "median"),
        training_fold_max=("standardized_coefficient_training_fold", "max"),
        training_fold_sd=("standardized_coefficient_training_fold", "std"),
        fraction_training_folds_positive=("standardized_coefficient_training_fold", lambda x: float((x > 0).mean()))).reset_index()
    coefficients = pd.DataFrame(full_coefficients).merge(coefficient_summary, on=["scenario", "response", "model", "feature"])
    coefficients.to_csv(OUT/"process_associations_coefficients.csv", index=False, encoding="utf-8-sig")
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUT/"process_associations_heldout_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = []
    for (scenario, response, model_name), part in pred.groupby(["scenario", "response", "model"]):
        metrics.append(dict(scenario=scenario, response=response, model=model_name, n_predictions=len(part),
                            grouped_loo_r2=r2_score(part.y_true, part.y_pred),
                            grouped_loo_rmse=np.sqrt(mean_squared_error(part.y_true, part.y_pred)),
                            grouped_loo_mae=mean_absolute_error(part.y_true, part.y_pred),
                            mean_baseline_grouped_loo_r2=r2_score(part.y_true, part.mean_baseline_prediction),
                            mean_baseline_grouped_loo_rmse=np.sqrt(mean_squared_error(part.y_true, part.mean_baseline_prediction))))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT/"process_associations_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(design_rows).to_csv(OUT/"process_input_design_diagnostics.csv", index=False, encoding="utf-8-sig")
    description = {
        "status": "completed_descriptive_analysis_and_source_group_holdouts",
        "responses": {"mean_CD": "Arithmetic mean of six CD Loss values in micrometres",
                      "mean_AR": "Arithmetic mean of six nominal AR values, dimensionless"},
        "scope": "These fixed composites summarize this layout only. They are not independently justified engineering acceptance functions, and structure-level findings remain necessary.",
        "models": {"OLS": "Unpenalized linear regression with fitted intercept", "Ridge_alpha1": "Ridge with alpha=1, fitted intercept; no outcome-based tuning"},
        "full_coefficients": "X and y standardized on the complete stated scenario, only to describe conditional associations. A coefficient is SD(y) per SD(X) conditional on the other included predictors.",
        "heldout_coefficients": "Each fold standardizes X and y on its training subset. The min/median/max range describes deletion sensitivity and changes in training scale; it is not a confidence interval or uncertainty about a physical causal effect.",
        "validation": "Leave one measurement-source group out; IDs 2 and 13 are kept together whenever both remain. All preprocessing is training-fold-only; R2 is computed from pooled original-unit held-out predictions, not averaged single-case R2.",
        "design_diagnostics": "Rank and condition number of standardized X and [1, standardized X]; raw-X condition is included only to expose dependence on units. Input correlation is descriptive, not proof of causal identifiability.",
        "limitations": [
            "No randomized-treatment causal claim is established. Unrecorded cycle/time/batch factors can confound associations.",
            "Removing records 24, 2 or 13 is a sensitivity scenario, not a declaration that the records are wrong.",
            "Mean AR changes its implicit geometry weighting because AR already divides by different design openings; the arithmetic mean does not mean average etch depth.",
            "Negative or near-zero held-out AR R2 prevents using coefficient signs as reliable mechanistic or engineering recommendations.",
            "Repeated measurements are unavailable, so a physical measurement-noise floor is not empirically identified.",
            "The record weights remain one per listed recipe, even where source independence of 2/13 is doubtful."
        ],
        "input_sha256": {str(SOURCE.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT/"process_associations_metadata.json").write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(coefficients[(coefficients.scenario=="without24") & (coefficients.response=="mean_CD")].to_string(index=False))


if __name__ == "__main__":
    main()
