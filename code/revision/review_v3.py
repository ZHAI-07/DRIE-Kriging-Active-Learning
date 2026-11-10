"""Review follow-up: tied query paths, exact signs, and original-protocol strata.

Reads saved, completed runs. No labels, kernels, seeds or budgets are changed.
Signs concern the fixed-data replay distribution, not new physical experiments.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/revision_v3/analysis"
SOURCES = {}
ATOL, RTOL = 1e-10, 1e-12


def source(rel):
    path = ROOT / rel
    SOURCES[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def csv(rel):
    return pd.read_csv(source(rel))


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    return frame


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    corrected = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    answer = np.empty(len(values)); answer[order] = corrected
    return answer


def signs(differences):
    values = np.asarray(differences, float)
    zero = np.isclose(values, 0., atol=ATOL, rtol=0.)
    wins, losses = int(np.sum(values < -ATOL)), int(np.sum(values > ATOL))
    n = wins + losses
    p = float(binomtest(wins, n, p=.5, alternative="two-sided").pvalue) if n else 1.
    # Independent elementary enumeration for p=.5, including the n=0 convention.
    reference = min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses)+1)) / 2**n) if n else 1.
    assert abs(p-reference) < 1e-12
    assert wins + losses + int(zero.sum()) == len(values)
    return dict(wins=wins, losses=losses, auc_ties=int(zero.sum()), n_nonzero=n, sign_p_two_sided=p)


def early():
    base = "outputs/revision/early_al/"
    auc, selected = csv(base+"auc.csv"), csv(base+"selections.csv")
    details, summaries = [], []
    for (scenario, kernel, budget), part in auc[auc.target == "normalized_mean"].groupby(["scenario", "kernel", "budget"]):
        pivot = part.pivot(index="seed", columns="strategy", values="test_auc").sort_index()
        assert len(pivot) == 30
        choices = selected[(selected.scenario == scenario) & (selected.kernel == kernel) & (selected.budget == budget)]
        for opponent in ["uncertainty", "maximin"]:
            rows = []
            for seed in pivot.index:
                group = choices[choices.seed == seed].sort_values("acquisition_step")
                a = group[group.strategy == "duw"].experiment_id.astype(int).tolist()
                b = group[group.strategy == opponent].experiment_id.astype(int).tolist()
                assert len(a) == len(b) == budget
                difference = float(pivot.loc[seed, "duw"] - pivot.loc[seed, opponent])
                same_path, same_set = a == b, set(a) == set(b)
                if same_path:
                    assert abs(difference) <= ATOL
                rows.append(dict(scenario=scenario, kernel=kernel, budget=int(budget), seed=int(seed),
                                 comparison="duw_minus_"+opponent, same_ordered_path=same_path,
                                 same_final_set=same_set, difference=difference,
                                 duw_path=" ".join(map(str, a)), other_path=" ".join(map(str, b))))
            details.extend(rows)
            d = pd.DataFrame(rows); discordant = d[~d.same_ordered_path]
            stats = signs(discordant.difference)
            summaries.append(dict(scenario=scenario, kernel=kernel, budget=int(budget), comparison="duw_minus_"+opponent,
                                  n_seeds=30, same_path_count=int(d.same_ordered_path.sum()),
                                  same_path_fraction=float(d.same_ordered_path.mean()),
                                  same_set_count=int(d.same_final_set.sum()), same_set_fraction=float(d.same_final_set.mean()),
                                  mean_delta_auc=float(d.difference.mean()), median_delta_auc=float(d.difference.median()),
                                  n_discordant_paths=len(discordant), **stats))
    summary = pd.DataFrame(summaries)
    # All 8 settings x both direct opponents are one exploratory family.
    summary["sign_p_holm_16"] = holm(summary.sign_p_two_sided)
    save("early_query_identity_per_seed", details)
    save("early_query_identity_and_signs", summary)
    return summary


def original_protocol():
    pair_rows, per_seed, trajectory_checks = [], [], []
    sys.path.insert(0, str(ROOT / "code"))
    from research.data import load_dataset
    for scenario, run in [("all24", "real_main"), ("without24", "real_drop24")]:
        folder = "outputs/results/"+run+"/"
        meta = json.loads(source(folder+"run_metadata.json").read_text(encoding="utf-8"))
        assert meta["status"] == "complete"
        cfg = meta["configuration"]
        assert cfg["experiment"]["n_test"] == cfg["experiment"]["n_initial"] == 6
        assert cfg["experiment"]["budget"] == 8 and cfg["experiment"]["repeats"] == 30
        assert cfg["exclude_ids"] == ([] if scenario == "all24" else [24])
        source_rel = "data/raw/user_confirmed_table_2026-11-09.json"
        assert hashlib.sha256(source(source_rel).read_bytes()).hexdigest() == meta["source_sha256"]
        for rel, digest in meta["code_sha256"].items():
            rel = rel.replace("\\", "/")
            assert hashlib.sha256(source(rel).read_bytes()).hexdigest() == digest
        auc, curves, selected = csv(folder+"al_auc.csv"), csv(folder+"al_curves.csv"), csv(folder+"al_selections.csv")
        splits = json.loads(source(folder+"al_splits.json").read_text(encoding="utf-8"))
        data = load_dataset(ROOT, "real", cfg["exclude_ids"])
        for split in splits:
            test, initial, pool = [set(split[k+"_ids"]) for k in ["test", "initial", "pool"]]
            assert len(test) == len(initial) == 6 and len(pool) == (12 if scenario == "all24" else 11)
            assert not test & (initial | pool)
            assert not ({2,13} & test and {2,13} & (initial | pool))
        errors = []
        for (seed, strategy, target), d in curves.groupby(["seed", "strategy", "target"]):
            d = d.sort_values("step"); assert list(d.step) == list(range(9))
            expected = float(np.trapezoid(d.rmse, d.step))
            actual = auc[(auc.seed == seed) & (auc.strategy == strategy) & (auc.target == target)].test_auc.iloc[0]
            errors.append(abs(expected - actual))
        assert max(errors) < 1e-8
        for target in ["__macro__", "__macro_cd__", "__macro_ar__"]:
            p = auc[auc.target == target].pivot(index="seed", columns="strategy", values="test_auc")
            assert len(p) == 30
            for opponent in ["random", "uncertainty", "maximin"]:
                delta = p.duw-p[opponent]
                stats = signs(delta)
                pair_rows.append(dict(scenario=scenario, run=run, target=target, comparison="duw_minus_"+opponent,
                                      n_seeds=30, mean_delta_auc=delta.mean(), median_delta_auc=delta.median(),
                                      sd_delta_auc=delta.std(ddof=1), **stats))
                per_seed.extend(dict(scenario=scenario, target=target, comparison="duw_minus_"+opponent,
                                     seed=int(seed), difference=float(value)) for seed, value in delta.items())
        trajectory_checks.append(dict(scenario=scenario, records=len(data.ids), n_targets=data.Y.shape[1],
                                      n_seeds=len(splits), curves=len(curves), queries=len(selected),
                                      max_auc_reconstruction_error=max(errors), code_and_source_hashes="matched"))
    save("original_protocol_direct_comparisons", pair_rows)
    save("original_protocol_per_seed", per_seed)
    return trajectory_checks


def aggregation():
    raw = json.loads(source("data/raw/user_confirmed_table_2026-11-09.json").read_text(encoding="utf-8-sig"))
    array = np.asarray(raw["rows"], float); rows=[]
    for scenario, keep in [("all24", np.ones(len(array), bool)), ("without24", array[:,0] != 24)]:
        ar = array[keep, 6::2]; means = ar.mean(axis=0)
        for j, structure in enumerate(raw["structures"]):
            rows.append(dict(scenario=scenario, structure=structure["label"], n_records=int(keep.sum()),
                             mean_AR=means[j], raw_mean_absolute_coefficient=1/6,
                             contribution_to_overall_raw_mean_fraction=means[j]/means.sum(),
                             change_in_raw_mean_for_one_percent_structure_change=.01*means[j]/6))
    save("raw_mean_AR_contributions", rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = early(); checks=original_protocol(); aggregation()
    result = dict(status="passed", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=SOURCES, original_protocol_checks=checks,
                  early_sign_comparisons=len(a), early_identity_mean_vs_uncertainty=float(a[a.comparison=="duw_minus_uncertainty"].same_path_fraction.mean()),
                  sign_policy={"null":"Among discordant paths with nonzero AUC difference: P(DUW wins)=0.5", "alternative":"two-sided",
                               "ties":"Identical ordered paths excluded; remaining |difference|<=1e-10 excluded from sign denominator",
                               "multiplicity":"Holm correction across all 16 exploratory early-budget direct comparisons",
                               "scope":"Conditional fixed-data replay distribution only; not new physical-experiment inference",
                               "independent_p_check":"Exact binomial probabilities independently enumerated with math.comb"},
                  statistical_sources=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
                                       "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html"],
                  no_models_refit=True, no_new_physical_experiments=True)
    (OUT/"validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","early_sign_comparisons","early_identity_mean_vs_uncertainty"]}))


if __name__ == "__main__":
    main()
