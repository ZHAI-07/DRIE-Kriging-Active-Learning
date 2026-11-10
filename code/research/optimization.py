"""Conservative optimization-readiness check and measured Pareto alternatives."""
from pathlib import Path
import numpy as np
import pandas as pd
from .experiments import frame_csv, write_json, clean_json


def nondominated(costs):
    """All columns are minimized; dominance requires at least one strict gain."""
    costs = np.asarray(costs, float)
    mask = np.ones(len(costs), dtype=bool)
    for i in range(len(costs)):
        mask[i] = not np.any(np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1))
    return mask


def optimization_readiness(dataset, out, basic_metrics):
    out = Path(out)
    if dataset.name != "real":
        write_json(out/"optimization_status.json", {
            "status": "not_targeted", "reason": "CFD is a separate algorithm-validation task; it does not supply DRIE recipe recommendations."})
        return
    costs = dataset.Y.copy()
    for j, metric in enumerate(dataset.metrics):
        if metric == "AR":
            costs[:, j] *= -1
    pareto = nondominated(costs)
    observed = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    observed.insert(0, "experiment_id", dataset.ids)
    observed["pareto_all_12_responses"] = pareto
    for j, target in enumerate(dataset.target_names):
        observed[target] = dataset.Y[:, j]
    frame_csv(observed, out/"observed_pareto.csv")
    checks = []
    if basic_metrics is not None:
        for target in dataset.target_names:
            p = basic_metrics[basic_metrics.target == target].set_index("model")
            gpr, baseline = p.loc["GPR"], p.loc["Mean"]
            checks.append({"target": target, "gpr_rmse": gpr.rmse,
                           "mean_rmse": baseline.rmse, "gpr_r2": gpr.r2,
                           "coverage95": gpr.coverage95,
                           "beats_mean": bool(gpr.rmse < baseline.rmse),
                           "positive_r2": bool(gpr.r2 > 0),
                           "coverage_screen": bool(gpr.coverage95 >= .75)})
    ready = bool(checks) and all(c["beats_mean"] and c["positive_r2"] and c["coverage_screen"] for c in checks)
    result = {"status": "continuous_recipe_recommendation_not_supported",
              "predictive_screen_passed": ready, "response_checks": checks,
              "screen_definition": "Predeclared conservative screen: every response beats its training-mean baseline, has pooled fold-out R2 > 0, and empirical 95% interval coverage >= 0.75. This heuristic screen is not proof of engineering validity.",
              "observed_pareto_ids": dataset.ids[pareto].tolist(),
              "pareto_definition": "Minimize all six CD Loss and maximize all six AR simultaneously; no undeclared averaging or weights.",
              "constraints": ["Raw-response source questions for experiments 2, 13 and 24 remain unresolved unless excluded in a named sensitivity run.",
                              "Equipment increments, fixed cycle/time settings and per-structure engineering acceptance limits need confirmation; four input units were user-confirmed.",
                              "No new physical etching experiment is available for verification."],
              "decision": "Report measured Pareto alternatives and prediction diagnostics. Do not manufacture a continuously optimized recipe or claim physical improvement.",
              "scope": "Measured alternatives in this finite table; not a continuous-domain optimum. Data-quality flags also apply to observed Pareto status."}
    write_json(out/"optimization_status.json", clean_json(result))
