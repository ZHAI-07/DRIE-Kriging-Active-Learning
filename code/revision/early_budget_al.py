"""Exploratory early-budget AL: two fixed structural averages, two GP kernels.

This reviewer-triggered check does not replace the frozen v1 results. Budget 2
and budget 4 are run independently (DUW weight depends on total budget), not
misrepresented truncations of a budget-8 run. No new physical data are created.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"code/revision"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from research.core import GPConfig, IndependentGP, safe_scale, select_next
from research.data import load_dataset, LabelOracle, make_split, source_groups
from research.experiments import frame_csv,write_json,clean_json
from model_comparison import IsotropicGP, append_mean_endpoints

OUT=ROOT/"outputs/revision/early_al"
STRATEGIES=("random","uncertainty","maximin","duw")
SCENARIOS={"all24":[],"without24":[24]}
CONFIG=GPConfig()
BUDGETS=(2,4)
SEEDS=list(range(20261109,20261139))


def composite_data(scenario):
    data=append_mean_endpoints(load_dataset(ROOT,"real",exclude_ids=SCENARIOS[scenario]))
    data.Y=data.Y[:,-2:]
    data.target_names=["mean_CD","mean_AR"]
    data.structures=["equal_weight_structures"]*2
    data.metrics=["CD Loss","AR"]
    return data


def seed_job(scenario,kernel,seed):
    data=composite_data(scenario)
    split=make_split(data,seed,6,6,max(BUDGETS))
    initial,pool,test=split["initial"],split["pool"],split["test"]
    groups=source_groups(data)
    assert not set(groups[test])&set(groups[np.r_[initial,pool]])
    scaler=StandardScaler().fit(data.X[initial])
    scaled_X=scaler.transform(data.X)
    evaluation_scale=safe_scale(data.Y[initial])
    factory=IndependentGP if kernel=="ARD" else IsotropicGP
    cache={};fit_records=[];curves=[];selections=[];splits=[]
    warning_counts=Counter()
    for budget in BUDGETS:
        splits.append({"scenario":scenario,"kernel":kernel,"budget":budget,"seed":seed,
                       "test_ids":data.ids[test].tolist(),"initial_ids":data.ids[initial].tolist(),
                       "pool_ids":data.ids[pool].tolist(),"initial_target_scale":evaluation_scale.tolist(),
                       "fixed_input_mean":scaler.mean_.tolist(),"fixed_input_scale":scaler.scale_.tolist(),
                       "record24_location":next((name for name,ind in [("test",test),("initial",initial),("pool",pool)] if 24 in data.ids[ind]),"excluded")})
        for strategy in STRATEGIES:
            oracle=LabelOracle(data,initial,pool,test)
            rng=np.random.default_rng(seed+918273)
            for step in range(budget+1):
                train_idx,X,Y=oracle.training()
                key=tuple(train_idx.tolist())
                if key not in cache:
                    gp=factory(CONFIG,seed=seed).fit(X,Y,x_scaler=scaler)
                    pred,std=gp.predict(data.X)
                    cache[key]=(pred,std,gp.y_scale.copy())
                    diag=gp.diagnostics()
                    for w in diag["warnings"]:warning_counts[w["message"]]+=1
                    fit_records.append({"training_ids":data.ids[train_idx].tolist(),
                                        "kernel_hyperparameters":diag["theta"],
                                        "optimizer_calls":len(diag["optimizer_runs"]),
                                        "optimizer_failures":sum(not x["success"] for x in diag["optimizer_runs"]),
                                        "warnings":len(diag["warnings"])})
                pred,std,current_scale=cache[key]
                # Test outcomes are read only after fitting, for scoring only.
                error=pred[test]-data.Y[test]
                rmse=np.sqrt(np.mean(error**2,axis=0));nr=rmse/evaluation_scale
                common={"scenario":scenario,"kernel":kernel,"budget":budget,"seed":seed,
                        "strategy":strategy,"step":step,"n_train":len(train_idx)}
                for j,target in enumerate(data.target_names):
                    curves.append({**common,"target":target,"rmse":rmse[j],"nrmse":nr[j],
                                   "coverage95":np.mean(abs(error[:,j])<=1.96*std[test,j])})
                curves.append({**common,"target":"normalized_mean","rmse":nr.mean(),"nrmse":nr.mean(),
                               "coverage95":np.mean(abs(error)<=1.96*std[test])})
                if step==budget:break
                candidates=np.array(sorted(oracle.candidates))
                chosen,scores=select_next(strategy,scaled_X[candidates],scaled_X[train_idx],
                    std[candidates]/current_scale,step,budget,rng,base=.2,decay=.6,penalty_mode="weighted")
                chosen_idx=int(candidates[chosen])
                selections.append({**common,"acquisition_step":step+1,
                                   "experiment_id":int(data.ids[chosen_idx]),
                                   "score":None if scores is None else float(scores[chosen])})
                oracle.reveal(chosen_idx)
    assert all(not set(f["training_ids"])&set(data.ids[test]) for f in fit_records)
    return clean_json({"scenario":scenario,"kernel":kernel,"seed":seed,"curves":curves,
                       "selections":selections,"splits":splits,"fits":fit_records,
                       "warning_counts":dict(warning_counts)})


def summarize(results):
    curves=pd.DataFrame([r for job in results for r in job["curves"]])
    selections=pd.DataFrame([r for job in results for r in job["selections"]])
    splits=[r for job in results for r in job["splits"]]
    grouping=["scenario","kernel","budget","seed","strategy","target"]
    auc=[]
    for keys,d in curves.groupby(grouping,sort=True):
        d=d.sort_values("step")
        auc.append({**dict(zip(grouping,keys)),"test_auc":float(np.trapezoid(d.rmse,d.step)),
                    "normalized_test_auc":float(np.trapezoid(d.nrmse,d.step)),
                    "final_rmse":float(d.rmse.iloc[-1]),"final_nrmse":float(d.nrmse.iloc[-1])})
    auc=pd.DataFrame(auc)
    summary=auc.groupby(["scenario","kernel","budget","strategy","target"],sort=True).agg(
        n_seeds=("seed","count"),mean_test_auc=("test_auc","mean"),sd_test_auc=("test_auc","std"),
        median_test_auc=("test_auc","median"),mean_final_rmse=("final_rmse","mean")).reset_index()
    pairs=[];overlaps=[]
    for keys,d in auc.groupby(["scenario","kernel","budget","target"],sort=True):
        pivot=d.pivot(index="seed",columns="strategy",values="test_auc")
        for comparison in ("random","uncertainty","maximin"):
            diff=(pivot.duw-pivot[comparison]).to_numpy()
            rng=np.random.default_rng(20261109)
            boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            pairs.append({**dict(zip(["scenario","kernel","budget","target"],keys)),
                          "comparison":"duw_minus_"+comparison,"n_seeds":len(diff),
                          "mean_delta_auc":diff.mean(),"median_delta_auc":np.median(diff),
                          "sd_delta_auc":np.std(diff,ddof=1),"q25_delta_auc":np.quantile(diff,.25),
                          "q75_delta_auc":np.quantile(diff,.75),"fraction_duw_better":np.mean(diff<0),
                          "fraction_exact_ties":np.mean(diff==0),
                          "mc_mean_interval_low":np.quantile(boot,.025),"mc_mean_interval_high":np.quantile(boot,.975),
                          "interval_scope":"fixed-data randomization expectation only; not physical population inference"})
    for keys,d in selections.groupby(["scenario","kernel","budget","seed"],sort=True):
        for comparison in ("uncertainty","maximin"):
            left=set(d[d.strategy=="duw"].experiment_id);right=set(d[d.strategy==comparison].experiment_id)
            overlaps.append({**dict(zip(["scenario","kernel","budget","seed"],keys)),
                             "comparison":"duw_vs_"+comparison,"acquired_only_jaccard":len(left&right)/len(left|right)})
    for name,frame in [("curves",curves),("selections",selections),("auc",auc),("summary",summary),
                       ("pairwise",pd.DataFrame(pairs)),("selection_overlap",pd.DataFrame(overlaps))]:
        frame_csv(frame,OUT/f"{name}.csv")
    write_json(OUT/"splits.json",splits)
    write_json(OUT/"fit_diagnostics.json",[{k:v for k,v in job.items() if k not in ["curves","selections","splits"]} for job in results])
    # Mechanical equality, budget and source-side checks on every replay.
    for split in splits:
        scenario,kernel,budget,seed=[split[k] for k in ["scenario","kernel","budget","seed"]]
        q=selections[(selections.scenario==scenario)&(selections.kernel==kernel)&(selections.budget==budget)&(selections.seed==seed)]
        initial=curves[(curves.scenario==scenario)&(curves.kernel==kernel)&(curves.budget==budget)&(curves.seed==seed)&(curves.step==0)]
        for target,d in initial.groupby("target"):assert np.ptp(d.rmse)<1e-12
        for strategy in STRATEGIES:
            ids=q[q.strategy==strategy].experiment_id.tolist()
            assert len(ids)==budget and len(set(ids))==budget and set(ids)<=set(split["pool_ids"])
            assert not set(ids)&set(split["test_ids"])
    return {"paired_configurations":8,"seeds_per_configuration":30,"strategy_replays":960,
            "curves":len(curves),"selections":len(selections),"budget_and_initial_equality_checks":"passed"}


def leakage_probe():
    data=composite_data("all24");split=make_split(data,SEEDS[0],6,6,4)
    initial=split["initial"];unseen=np.r_[split["pool"],split["test"]]
    scaler=StandardScaler().fit(data.X[initial]);Z=scaler.transform(data.X)
    results=[]
    for factory in [IndependentGP,IsotropicGP]:
        a=factory(CONFIG,seed=SEEDS[0]).fit(data.X[initial],data.Y[initial],scaler)
        altered=data.Y.copy();altered[unseen]=1e9*np.arange(len(unseen)*2).reshape(-1,2)
        b=factory(CONFIG,seed=SEEDS[0]).fit(data.X[initial],altered[initial],scaler)
        ma,sa=a.predict(data.X);mb,sb=b.predict(data.X)
        assert np.allclose(ma,mb) and np.allclose(sa,sb)
        for strategy in STRATEGIES[1:]:
            pa,_=select_next(strategy,Z[split["pool"]],Z[initial],sa[split["pool"]]/a.y_scale,0,4,np.random.default_rng(0))
            pb,_=select_next(strategy,Z[split["pool"]],Z[initial],sb[split["pool"]]/b.y_scale,0,4,np.random.default_rng(0))
            assert pa==pb
        results.append(factory.__name__)
    return {"hidden_label_poisoning_invariance":"passed","models":results}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),ROOT/"code/revision/model_comparison.py",ROOT/"code/research/core.py",ROOT/"code/research/data.py",ROOT/"data/raw/user_confirmed_table_2026-11-09.json"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol={"status":"running","design":"reviewer-triggered exploratory comparison, not independent confirmation",
              "scenarios":SCENARIOS,"kernels":["ARD","isotropic"],"budgets":BUDGETS,"seeds":SEEDS,
              "n_test":6,"n_initial":6,"gp":asdict(CONFIG),"source_sha256":hashes,
              "endpoints":{"mean_CD":"unweighted mean of six CD losses (micrometres)","mean_AR":"unweighted mean of six nominal AR values (dimensionless)"},
              "aggregation_scope":"fixed exploratory endpoint, not an established engineering utility; not mean per-structure RMSE",
              "UK_not_used":"5 trend coefficients with n_initial=6 gives at most one residual degree of freedom and occasional rank deficiency",
              "scaling":"input and evaluation scales from initial labels only; target fitting scale from current revealed training labels",
              "no_physical_experiments":True}
    write_json(OUT/"protocol.json",protocol)
    validation=leakage_probe();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs=[executor.submit(seed_job,scenario,kernel,seed) for scenario in SCENARIOS for kernel in ("ARD","isotropic") for seed in SEEDS]
        for i,future in enumerate(as_completed(jobs),1):
            results.append(future.result())
            if i%20==0:print(f"Completed {i}/{len(jobs)} seed jobs",flush=True)
    validation.update(summarize(results))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),"Inputs changed while running"
    write_json(OUT/"validation.json",validation)
    protocol["status"]="complete";write_json(OUT/"protocol.json",protocol)
    print(json.dumps(validation))


if __name__=="__main__":main()
