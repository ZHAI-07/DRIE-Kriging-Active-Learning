"""Recheck reviewer claims from frozen observations and saved v1 predictions.

This script never fits a GP, changes experimental labels, or selects a new
model. Bootstrap intervals quantify Monte Carlo partitions of this fixed
dataset only. All post-hoc stratification is labelled descriptive.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/revision/evidence"
RUN = ROOT / "outputs/results/real_main"
SOURCES: dict[str, str] = {}
BOOTSTRAP_REPEATS = 100000
BOOTSTRAP_SEED = 20261109


def source(path: Path) -> Path:
    SOURCES[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def save_csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return frame


def conditional_interval(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()


def mean_stats(values):
    values = np.asarray(values, float)
    q = np.quantile(values, [.025, .25, .5, .75, .975])
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)),
                q025=q[0], q25=q[1], median=q[2], q75=q[3], q975=q[4])


def expected_random_jaccard(pool_size, acquired, shared_initial=0):
    # Exact overlap law for two independent uniform subsets of a common pool.
    answer = 0.
    for overlap in range(max(0, 2*acquired-pool_size), acquired+1):
        prob = (math.comb(acquired, overlap)*math.comb(pool_size-acquired, acquired-overlap)
                / math.comb(pool_size, acquired))
        answer += prob*(shared_initial+overlap)/(shared_initial+2*acquired-overlap)
    return answer


def observation_checks(raw):
    a = np.asarray(raw["rows"], float)
    ids, Y = a[:, 0].astype(int), a[:, 5:]
    structures = [s["label"] for s in raw["structures"]]
    openings = np.array([s["design_opening"] for s in raw["structures"]], float)
    i24 = int(np.flatnonzero(ids == 24)[0]); other = ids != 24
    variation = []
    for j, label in enumerate(structures):
        y = Y[:, 2*j+1]; centered = y-y.mean(); sst = np.sum(centered**2)
        rest = y[other]; depth = y*openings[j]
        cd = Y[:, 2*j]
        variation.append(dict(
            structure=label, ar24=y[i24], ar_other_mean=rest.mean(), ar_other_median=np.median(rest),
            ar24_squared_deviation_fraction=centered[i24]**2/sst,
            leave24_sst_reduction_fraction=1.-np.sum((rest-rest.mean())**2)/sst,
            depth24_um=depth[i24], depth_other_mean_um=depth[other].mean(),
            depth_other_median_um=np.median(depth[other]),
            ratio_to_other_mean=y[i24]/rest.mean(), ratio_to_other_median=y[i24]/np.median(rest),
            other_min_pct_from_mean=100*(rest.min()/rest.mean()-1),
            other_max_pct_from_mean=100*(rest.max()/rest.mean()-1),
            cd24_um=cd[i24], cd_other_median_um=np.median(cd[other]),
            cd24_percentile_midrank=100*(np.sum(cd[other] < cd[i24])+.5*np.sum(cd[other] == cd[i24]))/len(rest)))
    variation = save_csv("sample24_response_leverage.csv", variation)
    corr_rows, pca_rows = [], []
    scenarios = {"all24": [], "without24": [24], "without24_and13": [24, 13], "without24_and2": [24, 2]}
    for scenario, omit in scenarios.items():
        keep = ~np.isin(ids, omit)
        for metric, offset in [("CD Loss", 0), ("AR", 1)]:
            corr = pd.DataFrame(Y[keep, offset::2], columns=structures).corr()
            corr.to_csv(OUT / f"correlation_{scenario}_{'cd' if offset==0 else 'ar'}.csv", encoding="utf-8-sig")
            lower = corr.values[np.tril_indices(6, -1)]
            corr_rows.append(dict(scenario=scenario, metric=metric, n_records=int(keep.sum()),
                                  pair_count=len(lower), min_r=lower.min(), median_r=np.median(lower), max_r=lower.max()))
        for scaling in ["raw_numeric_units", "column_standardized_descriptive"]:
            centered = Y[keep]-Y[keep].mean(axis=0)
            if scaling.startswith("column"):
                centered = centered/centered.std(axis=0, ddof=0)
            s = np.linalg.svd(centered, compute_uv=False)
            explained = s**2/np.sum(s**2)
            pca_rows.append(dict(scenario=scenario, scaling=scaling, n_records=int(keep.sum()),
                                 pc1_fraction=explained[0], pc2_fraction=explained[1],
                                 first2_fraction=explained[:2].sum(), first3_fraction=explained[:3].sum()))
    save_csv("response_correlations_summary.csv", corr_rows)
    save_csv("response_pca_summary.csv", pca_rows)
    inputs = a[:, 1:5]
    position = (inputs[i24]-inputs.min(axis=0))/np.ptp(inputs, axis=0)
    return dict(variance_fraction_min=float(variation.ar24_squared_deviation_fraction.min()),
                variance_fraction_max=float(variation.ar24_squared_deviation_fraction.max()),
                depth_ratio_mean_range=[float(variation.ratio_to_other_mean.min()), float(variation.ratio_to_other_mean.max())],
                sample24_observed_range_positions=position.tolist(),
                variance_formula="(y_24 - mean_all_24)^2 / sum_i (y_i - mean_all_24)^2",
                exclusion_sst_formula="1 - sum_i_not_24 (y_i - mean_other_23)^2 / SST_all_24; differs because the center changes",
                pca_note="Raw PCA mixes micrometre and dimensionless numerical scales. Standardized PCA is descriptive only, not a model transform or justification for data-driven response aggregation.")


def baseline_checks():
    basic = pd.read_csv(source(RUN/"basic_metrics.csv"))
    pivot = basic.pivot(index=["target", "structure", "metric"], columns="model", values="r2").reset_index()
    pivot["ridge_minus_gpr_r2"] = pivot.Ridge-pivot.GPR
    pivot["ridge_better"] = pivot["ridge_minus_gpr_r2"] > 1e-12
    pivot["gpr_better_than_mean"] = pivot.GPR > pivot.Mean+1e-12
    save_csv("baseline_r2_comparison.csv", pivot)
    return {"ridge_beats_gpr": int(pivot.ridge_better.sum()), "targets": len(pivot),
            "by_metric": {m: int(d.ridge_better.sum()) for m, d in pivot.groupby("metric")},
            "gpr_beats_mean": int(pivot.gpr_better_than_mean.sum())}


def paired_checks(auc):
    rows, seed_rows = [], []
    for target in ["__macro__", "__macro_cd__", "__macro_ar__"]:
        part = auc[auc.target == target]
        p = part.pivot(index="seed", columns="strategy", values="test_auc")
        for comparator in ["random", "uncertainty", "maximin"]:
            difference = p.duw-p[comparator]
            low, high = conditional_interval(difference)
            stats = mean_stats(difference)
            rows.append(dict(target=target, comparison=f"duw_minus_{comparator}", **stats,
                             mean_difference_mc_bootstrap_025=low, mean_difference_mc_bootstrap_975=high,
                             fraction_duw_better=float((difference < 0).mean()),
                             mean_paired_relative_improvement_pct=float((100*(p[comparator]-p.duw)/p[comparator]).mean()),
                             fraction_exact_ties=float((difference == 0).mean()),
                             interval_scope="Conditional on this finite dataset and split/selection protocol only; no physical-population inference"))
            for seed, value in difference.items():
                seed_rows.append(dict(seed=int(seed), target=target, comparison=f"duw_minus_{comparator}", difference=value))
    result = save_csv("duw_pairwise_comparisons.csv", rows)
    save_csv("duw_pairwise_per_seed.csv", seed_rows)
    return result[result.target == "__macro__"].to_dict(orient="records")


def selection_checks(selections, splits):
    detailed = []
    for split in splits:
        seed = split["seed"]; initial = set(split["initial_ids"])
        part = selections[selections.seed == seed].sort_values("acquisition_step")
        sequence = {s: p.experiment_id.astype(int).tolist() for s, p in part.groupby("strategy")}
        for budget in [1, 2, 4, 8]:
            for left, right in itertools.combinations(["random", "uncertainty", "maximin", "duw"], 2):
                a, b = set(sequence[left][:budget]), set(sequence[right][:budget])
                detailed.append(dict(seed=seed, budget=budget, left=left, right=right,
                                     acquired_only_jaccard=len(a & b)/len(a | b),
                                     with_shared_initial_jaccard=len((a | initial) & (b | initial))/len((a | initial) | (b | initial)),
                                     identical_ordered_prefix=sequence[left][:budget] == sequence[right][:budget],
                                     random_pair_expected_acquired_only_jaccard=expected_random_jaccard(len(split["pool_ids"]), budget),
                                     random_pair_expected_with_initial_jaccard=expected_random_jaccard(len(split["pool_ids"]), budget, len(initial))))
    detailed = save_csv("selection_overlap_per_seed.csv", detailed)
    keys = ["budget", "left", "right"]
    summary = detailed.groupby(keys).agg(
        n_seeds=("seed", "nunique"), mean_acquired_only_jaccard=("acquired_only_jaccard", "mean"),
        min_acquired_only_jaccard=("acquired_only_jaccard", "min"), max_acquired_only_jaccard=("acquired_only_jaccard", "max"),
        mean_with_shared_initial_jaccard=("with_shared_initial_jaccard", "mean"),
        identical_ordered_prefix_fraction=("identical_ordered_prefix", "mean"),
        random_pair_expected_acquired_only_jaccard=("random_pair_expected_acquired_only_jaccard", "first"),
        random_pair_expected_with_initial_jaccard=("random_pair_expected_with_initial_jaccard", "first")).reset_index()
    save_csv("selection_overlap_summary.csv", summary)
    return summary[(summary.left == "uncertainty") & (summary.right == "duw")].to_dict(orient="records")


def stratification_checks(auc, selections, splits):
    locations = {}
    for split in splits:
        found = [name for name in ["test", "initial", "pool"] if 24 in split[name+"_ids"]]
        assert len(found) == 1
        locations[split["seed"]] = found[0]
    choices = selections.groupby(["seed", "strategy"]).apply(
        lambda g: g.loc[g.experiment_id == 24, "acquisition_step"].iloc[0] if (g.experiment_id == 24).any() else np.nan,
        include_groups=False).to_dict()
    rows, detailed = [], []
    for target in ["__macro__", "__macro_cd__", "__macro_ar__"]:
        p = auc[auc.target == target].pivot(index="seed", columns="strategy", values="test_auc")
        for location in ["test", "initial", "pool", "not_pool"]:
            seeds = [seed for seed, loc in locations.items() if loc == location or (location == "not_pool" and loc != "pool")]
            for strategy in ["random", "uncertainty", "maximin", "duw"]:
                vals = p.loc[seeds, strategy]
                delta = vals-p.loc[seeds, "random"]
                acquisition_steps = [choices[(seed, strategy)] for seed in seeds]
                selected = np.isfinite(acquisition_steps)
                rows.append(dict(target=target, sample24_location=location, strategy=strategy,
                                 n_seeds=len(seeds), mean_test_auc=float(vals.mean()), sd_test_auc=float(vals.std(ddof=1)),
                                 mean_auc_difference_vs_random=float(delta.mean()),
                                 median_auc_difference_vs_random=float(delta.median()),
                                 mean_relative_improvement_pct=float((100*(p.loc[seeds, "random"]-vals)/p.loc[seeds, "random"]).mean()),
                                 sample24_acquisition_fraction=float(selected.mean()) if location == "pool" else np.nan,
                                 mean_acquisition_step_if_selected=float(np.asarray(acquisition_steps)[selected].mean()) if selected.any() else np.nan,
                                 interpretation="Post-hoc descriptive subset of the same data and seeds. Not a causal experiment or a prespecified subgroup test."))
        for seed, values in p.iterrows():
            for strategy in ["random", "uncertainty", "maximin", "duw"]:
                detailed.append(dict(seed=int(seed), target=target, sample24_location=locations[seed], strategy=strategy,
                                     test_auc=values[strategy], difference_vs_random=values[strategy]-values["random"],
                                     acquisition_step=choices[(seed, strategy)]))
    out = save_csv("sample24_strata_recomputed.csv", rows)
    save_csv("sample24_strata_per_seed.csv", detailed)
    return out[(out.target == "__macro__") & (out.strategy.isin(["random", "duw"]))].to_dict(orient="records")


def prefix_performance_checks(auc):
    curves = pd.read_csv(source(RUN/"al_curves.csv"))
    targets = ["__macro__", "__macro_cd__", "__macro_ar__"]
    detailed = []
    for (seed, strategy, target), frame in curves[curves.target.isin(targets)].groupby(["seed", "strategy", "target"]):
        frame = frame.sort_values("step")
        for budget in [1, 2, 4, 8]:
            prefix = frame[frame.step <= budget]
            assert len(prefix) == budget+1
            area = float(np.trapezoid(prefix.rmse, x=prefix.step))
            if budget == 8:
                old_area = auc[(auc.seed == seed) & (auc.strategy == strategy) & (auc.target == target)].test_auc.iloc[0]
                assert np.isclose(area, old_area, rtol=1e-12, atol=1e-12)
            detailed.append(dict(seed=seed, strategy=strategy, target=target, prefix_budget=budget,
                                 prefix_test_auc=area, endpoint_rmse=float(prefix.rmse.iloc[-1])))
    detailed = save_csv("early_prefix_performance_per_seed.csv", detailed)
    rows = []
    for (budget, target), frame in detailed.groupby(["prefix_budget", "target"]):
        table = frame.pivot(index="seed", columns="strategy", values="prefix_test_auc")
        for comparator in ["random", "uncertainty", "maximin"]:
            difference = table.duw-table[comparator]
            low, high = conditional_interval(difference)
            rows.append(dict(prefix_budget=budget, target=target, comparison=f"duw_minus_{comparator}",
                             **mean_stats(difference), mean_difference_mc_bootstrap_025=low,
                             mean_difference_mc_bootstrap_975=high,
                             fraction_duw_better=float((difference < 0).mean()),
                             mean_paired_relative_improvement_pct=float((100*(table[comparator]-table.duw)/table[comparator]).mean()),
                             interpretation="Post-hoc prefixes of the ORIGINAL K=8 acquisition schedule. These are not independently rerun K=1/2/4 policies; DUW weight decay still uses K=8."))
    save_csv("early_prefix_pairwise_comparisons.csv", rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(source(ROOT/"data/raw/user_confirmed_table_2026-11-09.json").read_text(encoding="utf-8-sig"))
    auc = pd.read_csv(source(RUN/"al_auc.csv"))
    selections = pd.read_csv(source(RUN/"al_selections.csv"))
    splits = json.loads(source(RUN/"al_splits.json").read_text(encoding="utf-8"))
    summary = {
        "status": "completed_from_saved_sources_without_new_GP_runs",
        "data_leverage": observation_checks(raw), "baseline": baseline_checks(),
        "duw_macro_pairwise": paired_checks(auc),
        "duw_uncertainty_overlap_by_budget": selection_checks(selections, splits),
        "sample24_macro_strata": stratification_checks(auc, selections, splits),
        "bootstrap_repeats": BOOTSTRAP_REPEATS, "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation_corrections": [
            "The dominance of record 24 and stratified acquisition associations are directly measurable. A different etching time/cycle count is a plausible but unverified physical explanation, not an established exclusion criterion.",
            "Identical labels for different inputs are mathematically compatible with a non-injective response function. The evidence against independence of 2/13 is the shared original measurements and byte-identical images; the trustworthy record cannot be identified from prediction scores.",
            "GP predictive covariance is label-independent conditional on the kernel/noise hyperparameters. Here fitted hyperparameters and output scales depend on revealed training labels, so uncertainty sampling is not strictly a fixed geometric rule.",
            "Additional split seeds reduce Monte Carlo error for the finite-data randomization expectation, but do not create additional physical observations or establish population-level benefits.",
            "High output correlations do not authorize an undeclared engineering composite. If a composite is introduced, its physical meaning, weighting and training-only estimation must be fixed and structure-specific results retained.",
            "Two PCs explaining most variance is scale-dependent. Raw micrometre and dimensionless responses must not be pooled without explaining the chosen metric.",
            "The source status of the CFD output does not establish that it is noise-free or deterministic, and space-filling dominance over random is not a mathematical necessity for every response/candidate budget.",
            "Selection overlap must distinguish the acquired subset from the complete labeled set, because the initial six records are shared by construction. The large budget also raises chance overlap."
        ],
    }
    summary["source_sha256"] = SOURCES
    prefix_performance_checks(auc)
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # JSON has no NaN: absent acquisition statistics are null.
    content = json.loads(json.dumps(summary, ensure_ascii=False, default=float).replace("NaN", "null"))
    (OUT/"review_evidence_summary.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "variance": summary["data_leverage"], "baseline": summary["baseline"],
                      "pairwise": summary["duw_macro_pairwise"], "overlap": summary["duw_uncertainty_overlap_by_budget"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
