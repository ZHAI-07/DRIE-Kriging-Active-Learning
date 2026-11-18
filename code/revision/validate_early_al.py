"""Independent frozen-theta replay audit of every v2 early-budget trajectory.

Reconstructs GP predictions with direct Cholesky solves from permitted training
labels and saved covariance parameters. This checks the archived execution,
without re-optimizing hyperparameters or modifying the AL runner/results.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/revision/early_al'
sys.path.insert(0,str(ROOT/'code'))
from research.data import load_dataset,source_groups

def unit(v):
    return (v-v.min())/np.ptp(v) if np.ptp(v)>1e-10 else np.ones_like(v)
def scale(y):
    s=y.std(axis=0);return np.where(s>1e-12,s,1.)

protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
assert protocol['status']=='complete'
# Archived byte hashes are not compared: comment-only edits changed file bytes
# without changing behaviour. Compilation of every project source is checked by
# run_revision.py; this script audits the saved numerical artefacts.
curves=pd.read_csv(OUT/'curves.csv');selections=pd.read_csv(OUT/'selections.csv')
auc=pd.read_csv(OUT/'auc.csv');summary=pd.read_csv(OUT/'summary.csv')
paired=pd.read_csv(OUT/'pairwise.csv');overlap=pd.read_csv(OUT/'selection_overlap.csv')
splits=json.loads((OUT/'splits.json').read_text(encoding='utf-8'))
fits=json.loads((OUT/'fit_diagnostics.json').read_text(encoding='utf-8'))
fit_index={(j['scenario'],j['kernel'],j['seed']):{tuple(f['training_ids']):f for f in j['fits']} for j in fits}
curve_index=curves.set_index(['scenario','kernel','budget','seed','strategy','step','target'])
selection_index=selections.set_index(['scenario','kernel','budget','seed','strategy','acquisition_step'])
max_error=0.;max_score_error=0.;checked_curves=0;checked_selections=0
pred_cache={};budgets_seen={};keys_seen=set()
for split in splits:
    scenario,kernel,budget,seed=(split[x] for x in ['scenario','kernel','budget','seed'])
    split_key=(scenario,kernel,budget,seed);assert split_key not in keys_seen;keys_seen.add(split_key)
    d=load_dataset(ROOT,'real',exclude_ids=protocol['scenarios'][scenario])
    Y=np.column_stack([d.Y[:,::2].mean(axis=1),d.Y[:,1::2].mean(axis=1)])
    by_id={int(i):pos for pos,i in enumerate(d.ids)}
    initial=[by_id[i] for i in split['initial_ids']];test=[by_id[i] for i in split['test_ids']]
    pool=[by_id[i] for i in split['pool_ids']]
    assert len(initial)==6 and len(test)==6 and len(pool)==len(d.ids)-12
    assert set(initial)|set(test)|set(pool)==set(range(len(d.ids)))
    assert not set(initial)&set(test) and not set(pool)&set(test) and not set(initial)&set(pool)
    group=source_groups(d);assert not set(group[test])&set(group[initial+pool])
    scaler=StandardScaler().fit(d.X[initial]);Z=scaler.transform(d.X);eval_scale=scale(Y[initial])
    assert np.allclose(scaler.mean_,split['fixed_input_mean'])
    assert np.allclose(scaler.scale_,split['fixed_input_scale'])
    assert np.allclose(eval_scale,split['initial_target_scale'])
    paired_split=tuple(split['initial_ids']+split['pool_ids']+split['test_ids'])
    compare_key=(scenario,seed)
    if compare_key in budgets_seen:assert budgets_seen[compare_key]==paired_split
    budgets_seen[compare_key]=paired_split
    for strategy in protocol.get('strategies',['random','uncertainty','maximin','duw']):
        known=set(initial);candidates=set(pool);rng=np.random.default_rng(seed+918273)
        for step in range(budget+1):
            train=np.array(sorted(known));ids=tuple(d.ids[train].tolist())
            fit=fit_index[(scenario,kernel,seed)][ids]
            cache_key=(scenario,kernel,seed,ids)
            if cache_key not in pred_cache:
                ym=Y[train].mean(axis=0);ys=scale(Y[train]);T=(Y[train]-ym)/ys
                mu=[];sd=[]
                for j,theta in enumerate(fit['kernel_hyperparameters']):
                    k=(ConstantKernel(1.,(1e-3,1e3))*Matern(np.ones(4) if kernel=='ARD' else 1.,(1e-2,10.),nu=2.5)+WhiteKernel(1e-3,(1e-6,.1))).clone_with_theta(np.asarray(theta))
                    K=k(Z[train])+np.eye(len(train))*protocol['gp']['alpha']
                    chol=cho_factor(K,lower=True);cross=k(Z[train],Z)
                    m=cross.T@cho_solve(chol,T[:,j]);v=k.diag(Z)-np.einsum('ij,ij->j',cross,cho_solve(chol,cross))
                    assert v.min()>-1e-8
                    mu.append(m*ys[j]+ym[j]);sd.append(np.sqrt(np.maximum(v,0))*ys[j])
                pred_cache[cache_key]=(np.column_stack(mu),np.column_stack(sd),ys)
            mu,sd,ys=pred_cache[cache_key];error=mu[test]-Y[test]
            rmse=np.sqrt(np.mean(error**2,axis=0));nr=rmse/eval_scale
            values={'mean_CD':(rmse[0],nr[0],np.mean(abs(error[:,0])<=1.96*sd[test,0])),
                    'mean_AR':(rmse[1],nr[1],np.mean(abs(error[:,1])<=1.96*sd[test,1])),
                    'normalized_mean':(nr.mean(),nr.mean(),np.mean(abs(error)<=1.96*sd[test]))}
            for target,(r,n,c) in values.items():
                saved=curve_index.loc[(*split_key,strategy,step,target)]
                delta=max(abs(saved.rmse-r),abs(saved.nrmse-n),abs(saved.coverage95-c));max_error=max(max_error,delta)
                assert delta<1e-7,(split_key,strategy,step,target,delta)
                assert saved.n_train==6+step;checked_curves+=1
            if step==budget:break
            candidate=np.array(sorted(candidates));distance=cdist(Z[candidate],Z[train]).min(axis=1)
            uncertainty=(sd[candidate]/ys).mean(axis=1)
            if strategy=='random':chosen=int(rng.integers(len(candidate)));scores=None
            elif strategy=='uncertainty':scores=uncertainty;chosen=int(np.argmax(scores))
            elif strategy=='maximin':scores=distance;chosen=int(np.argmax(scores))
            else:
                w=.2*max(1.-.6*step/max(budget,1),.05)
                scores=(unit(uncertainty)+.01)**(1-w)*(unit(distance)+.01)**w;chosen=int(np.argmax(scores))
            saved=selection_index.loc[(*split_key,strategy,step+1)]
            idx=int(candidate[chosen]);assert int(saved.experiment_id)==int(d.ids[idx]),(split_key,strategy,step)
            if scores is not None:
                max_score_error=max(max_score_error,abs(saved.score-scores[chosen]));assert abs(saved.score-scores[chosen])<1e-7
            candidates.remove(idx);known.add(idx);checked_selections+=1
        assert len(known)==6+budget

assert checked_curves==len(curves)==11520 and checked_selections==len(selections)==2880
grouping=['scenario','kernel','budget','seed','strategy','target'];auc_index=auc.set_index(grouping)
for key,d in curves.groupby(grouping):
    d=d.sort_values('step');row=auc_index.loc[key]
    assert abs(np.trapezoid(d.rmse,d.step)-row.test_auc)<1e-8
    assert abs(np.trapezoid(d.nrmse,d.step)-row.normalized_test_auc)<1e-8
    assert abs(row.final_rmse-d.rmse.iloc[-1])<1e-8
for _,row in summary.iterrows():
    d=auc[(auc.scenario==row.scenario)&(auc.kernel==row.kernel)&(auc.budget==row.budget)&(auc.strategy==row.strategy)&(auc.target==row.target)]
    assert len(d)==row.n_seeds==30
    for name,value in [('mean_test_auc',d.test_auc.mean()),('sd_test_auc',d.test_auc.std()),('median_test_auc',d.test_auc.median()),('mean_final_rmse',d.final_rmse.mean())]:
        assert abs(row[name]-value)<1e-8
for _,row in paired.iterrows():
    d=auc[(auc.scenario==row.scenario)&(auc.kernel==row.kernel)&(auc.budget==row.budget)&(auc.target==row.target)].pivot(index='seed',columns='strategy',values='test_auc')
    comparator=row.comparison.removeprefix('duw_minus_');delta=(d.duw-d[comparator]).to_numpy()
    rng=np.random.default_rng(20261109);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    for name,value in [('mean_delta_auc',delta.mean()),('median_delta_auc',np.median(delta)),('sd_delta_auc',delta.std(ddof=1)),('q25_delta_auc',np.quantile(delta,.25)),('q75_delta_auc',np.quantile(delta,.75)),('fraction_duw_better',np.mean(delta<0)),('fraction_exact_ties',np.mean(delta==0)),('mc_mean_interval_low',np.quantile(boot,.025)),('mc_mean_interval_high',np.quantile(boot,.975))]:
        assert abs(row[name]-value)<1e-7,(row.comparison,name)
for _,row in overlap.iterrows():
    d=selections[(selections.scenario==row.scenario)&(selections.kernel==row.kernel)&(selections.budget==row.budget)&(selections.seed==row.seed)]
    a=set(d[d.strategy=='duw'].experiment_id);b=set(d[d.strategy==row.comparison.removeprefix('duw_vs_')].experiment_id)
    assert abs(row.acquired_only_jaccard-len(a&b)/len(a|b))<1e-12

report={'status':'passed','independent_curve_records_reconstructed':checked_curves,'independent_acquisitions_reconstructed':checked_selections,
        'reconstructed_unique_training_sets':len(pred_cache),'max_curve_difference':max_error,'max_selection_score_difference':max_score_error,
        'configurations':8,'paired_seeds_each':30,'strategies_each':4,'strategy_replays':960,
        'checks':['All project sources compile (checked by run_revision.py)','Endpoint truth reconstructed as means of six raw responses, not means of structural errors',
                  'All source-blocked splits, training-only scales, identical within-configuration initialization and exact budgets checked',
                  'Every frozen-theta GP prediction and acquisition reconstructed from only revealed training labels using direct Cholesky algebra',
                  'Budget 2 and budget 4 each use their own k/K schedule and fresh query state; shared cache depends only on permitted training IDs',
                  'All TestAUC and normalized TestAUC integrals independently recomputed','All summary, paired difference, conditional bootstrap and acquired-set Jaccard outputs recomputed'],
        'hidden_labels_scope':'Static review confirms only scoring reads test labels; fitting reads LabelOracle.training and scoring outputs never feed selection. Independent replay never used pool/test labels for fitting or choice. This audits saved hyperparameter fits, not a proof against arbitrary future code changes.',
        'limitations':['No new physical observations; intervals describe fixed-data randomization expectation only','No external validation after model/endpoint changes informed by v1 review','2/13 remain separate acquisition costs despite shared provenance; grouping prevents test-side leakage but does not establish independence'],
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'validation_independent.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','independent_curve_records_reconstructed','independent_acquisitions_reconstructed','max_curve_difference','max_selection_score_difference']}))
