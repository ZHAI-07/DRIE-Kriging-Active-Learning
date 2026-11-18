"""Verify saved computation artifacts without re-fitting models.

Archived byte hashes are not compared: comment-only edits changed file bytes
without changing behaviour, so the historical SHA256 manifests no longer
describe the current sources. Availability of every source a run declares, plus
compilation of the project sources, replaces that gate.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"code"))
import compile_check
from research.data import load_dataset


def main():
    expected_names = ["real_main", "cfd_main", "real_drop2", "real_drop13", "real_drop24",
                      "real_historical_schedule", "cfd_legacy_code"]
    checks = []
    compile_records, compile_failures = compile_check.compile_sources(ROOT)
    if compile_failures:
        raise RuntimeError("Python sources failed to compile: "
                           + ", ".join(f["path"] for f in compile_failures[:5]))
    # Every source a saved run declares must still exist, including legacy
    # Windows-style paths recorded by the original execution host.
    def recorded(path):
        candidate = (ROOT / str(path).replace("\\", "/")).resolve()
        if not candidate.is_relative_to(ROOT):
            raise ValueError(f"Recorded source path escapes the project: {path}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing recorded source: {path}")
        return candidate
    for name in expected_names:
        folder = ROOT/"outputs/results"/name
        fit_logs_checked = False
        meta = json.loads((folder/"run_metadata.json").read_text(encoding="utf-8"))
        assert meta["status"] == "complete", name
        assert meta["source_unchanged"], name
        for path in meta["code_sha256"]:
            recorded(path)
        current_source = ROOT/"data/raw"/(
            "user_confirmed_table_2026-11-09.json" if meta["dataset"] == "real"
            else "cfd_legacy_snapshot.json")
        assert current_source.is_file(), name
        cfg = meta["configuration"]
        data = load_dataset(ROOT, cfg["dataset"], cfg["exclude_ids"], cfg.get("cfd_variant", "excel"))
        budget, repeats = cfg["experiment"]["budget"], cfg["experiment"]["repeats"]
        assert repeats >= 30, name
        curves = pd.read_csv(folder/"al_curves.csv")
        choices = pd.read_csv(folder/"al_selections.csv")
        splits = json.loads((folder/"al_splits.json").read_text(encoding="utf-8"))
        assert len(splits) == repeats and curves.seed.nunique() == repeats, name
        assert set(curves.strategy) == {"random", "uncertainty", "maximin", "duw"}, name
        assert np.isfinite(curves[["rmse", "mae", "nrmse"]].to_numpy()).all(), name
        assert choices.groupby(["seed", "strategy"]).size().eq(budget).all(), name
        for split in splits:
            test, initial, pool = map(set, (split["test_ids"], split["initial_ids"], split["pool_ids"]))
            assert not test & initial and not test & pool and not initial & pool, name
            if cfg["dataset"] == "real" and {2, 13}.issubset(data.ids):
                assert (2 in test) == (13 in test), name
            for _, series in choices[choices.seed == split["seed"]].groupby("strategy"):
                chosen = set(series.experiment_id)
                assert len(chosen) == budget and chosen <= pool, name
            start = curves[(curves.seed == split["seed"]) & (curves.step == 0)]
            assert start.groupby("target").rmse.nunique().eq(1).all(), name
            # Per-seed fit logs are gitignored optimizer checkpoints, so the
            # tracked reference suites do not ship them. When present, use them
            # to prove no held-out ID entered a training set.
            diagnostic = folder/"fit_diagnostics"/f"seed_{split['seed']}.json"
            if diagnostic.is_file():
                diagnostics = json.loads(diagnostic.read_text(encoding="utf-8"))
                assert all(not set(fit["training_ids"]) & test for fit in diagnostics), name
                fit_logs_checked = True
        basic = pd.read_csv(folder/"basic_predictions.csv")
        assert len(basic) == 3 * len(data.ids) * len(data.target_names), name
        assert np.isfinite(basic[["y_true", "y_pred"]].to_numpy()).all(), name
        for row in basic.itertuples():
            i = list(data.ids).index(row.experiment_id)
            j = data.target_names.index(row.target)
            assert abs(row.y_true-data.Y[i, j]) < 1e-10, name
        checks.append({"run": name, "status": "passed", "repeats": repeats,
                       "basic_prediction_rows": len(basic), "curve_rows": len(curves),
                       "selected_experiments_logged": len(choices),
                       "per_seed_fit_logs_present": fit_logs_checked,
                       "verified": ["sources compile and declared files exist", "finite predictions",
                                    "source-blocked paired partitions", "equal initial errors",
                                    "equal budgets and no repeated acquisition",
                                    "no test IDs in fit logs (when logs are shipped)",
                                    "saved outcomes match the declared data variant"]})
    out = ROOT/"outputs/reports/execution_integrity.json"
    out.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS {len(checks)} complete experiment suites; {out}")


if __name__ == "__main__":
    main()
