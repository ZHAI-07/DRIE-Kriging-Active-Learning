"""Descriptive sample-24 and optimizer diagnostics from saved run artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    results, diagnostics = [], []
    for folder in sorted((ROOT/"outputs/results").iterdir()):
        if not folder.is_dir() or not (folder/"run_metadata.json").exists():
            continue
        meta = json.loads((folder/"run_metadata.json").read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            continue
        record = {"run_name": folder.name, "target_gp_fits": 0, "optimizer_calls": 0,
                  "unsuccessful_optimizer_calls": 0, "warning_messages": 0}
        paths = list((folder/"fit_diagnostics").glob("*.json"))
        if (folder/"basic_fit_diagnostics.json").exists():
            paths.append(folder/"basic_fit_diagnostics.json")
        for path in paths:
            fits = json.loads(path.read_text(encoding="utf-8"))
            for fit in fits:
                record["target_gp_fits"] += len(fit["kernels"])
                record["optimizer_calls"] += len(fit["optimizer_runs"])
                record["unsuccessful_optimizer_calls"] += sum(not r["success"] for r in fit["optimizer_runs"])
                record["warning_messages"] += len(fit["warnings"])
        diagnostics.append(record)
        if meta["dataset"] != "real" or not (folder/"al_splits.json").exists():
            continue
        split = json.loads((folder/"al_splits.json").read_text(encoding="utf-8"))
        locations = {r["seed"]: r["flagged_id_locations"]["24"] for r in split}
        auc = pd.read_csv(folder/"al_auc.csv")
        selected = pd.read_csv(folder/"al_selections.csv")
        for target in ("__macro__", "__macro_cd__", "__macro_ar__"):
            wide = auc[auc.target == target].pivot(index="seed", columns="strategy", values="test_auc")
            wide["sample24_location"] = wide.index.map(locations)
            for location, part in wide.groupby("sample24_location"):
                for strategy in ("random", "uncertainty", "maximin", "duw"):
                    difference = part[strategy]-part["random"]
                    selection_rows = selected[(selected.strategy == strategy) &
                                              selected.seed.isin(part.index) &
                                              (selected.experiment_id == 24)]
                    results.append({"run_name": folder.name, "target": target,
                                    "strategy": strategy, "sample24_location": location,
                                    "n_seeds": len(part), "mean_test_auc": part[strategy].mean(),
                                    "mean_auc_difference_vs_random": difference.mean(),
                                    "mean_relative_improvement_percent": (-difference/part["random"]*100).mean(),
                                    "sample24_acquisition_fraction": selection_rows.seed.nunique()/len(part) if location == "pool" else np.nan,
                                    "mean_acquisition_step_if_selected": selection_rows.acquisition_step.mean() if len(selection_rows) else np.nan,
                                    "interpretation": "Post-hoc descriptive strata on fixed data; not causal evidence or an independently powered significance test."})
    out = ROOT/"outputs/reports"
    pd.DataFrame(results).to_csv(out/"sample24_descriptive_strata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(out/"optimizer_diagnostics_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Summarized {len(diagnostics)} completed runs; sample-24 descriptive rows: {len(results)}")


if __name__ == "__main__":
    main()
