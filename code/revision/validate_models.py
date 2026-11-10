"""Audit v2 saved model outputs, legacy agreement and independent linear solves."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(ROOT/"code/revision"))
from research.data import load_dataset,source_groups
from model_comparison import SCENARIOS,append_mean_endpoints

OUT=ROOT/"outputs/revision/models"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];summaries=[]
legacy={"real_all24":"real_main","real_without24":"real_drop24","cfd_excel":"cfd_main","cfd_legacy":"cfd_legacy_code"}
for name,(kind,exclude,variant) in SCENARIOS.items():
    folder=OUT/name
    meta=json.loads((folder/"run_metadata.json").read_text(encoding="utf-8"))
    assert meta['status']=='complete'
    assert meta['code_sha256']==sha(ROOT/'code/revision/model_comparison.py')
    # Archived manifests were produced on Windows; normalize separators only.
    assert meta['source_sha256']==sha(ROOT/meta['source'].replace('\\','/'))
    data=append_mean_endpoints(load_dataset(ROOT,kind,exclude,variant));groups=source_groups(data)
    pred=pd.read_csv(folder/'predictions.csv');metrics=pd.read_csv(folder/'metrics.csv')
    diagnostics=json.loads((folder/'fit_diagnostics.json').read_text(encoding='utf-8'))
    for d in diagnostics:
        assert not set(d['train_ids'])&set(d['test_ids'])
        assert len(d['train_ids'])+len(d['test_ids'])==len(data.ids)
        if kind=='real' and 2 in data.ids and 13 in data.ids:
            assert (2 in d['test_ids'])==(13 in d['test_ids'])
    assert np.isfinite(pred[['y_true','y_pred']]).all().all()
    assert not pred.duplicated(['model','target','experiment_id']).any()
    assert len(pred)==len(meta['models'])*len(data.ids)*data.Y.shape[1]
    for _,row in pred.iterrows():
        i=np.flatnonzero(data.ids==row.experiment_id)[0];j=data.target_names.index(row.target)
        assert row.y_true==data.Y[i,j] or abs(row.y_true-data.Y[i,j])<1e-12
    for _,row in metrics.iterrows():
        p=pred[(pred.model==row.model)&(pred.target==row.target)]
        assert abs(row.rmse-np.sqrt(np.mean((p.y_pred-p.y_true)**2)))<1e-10
        assert abs(row.r2-r2_score(p.y_true,p.y_pred))<1e-10
    max_linear_error=0.
    for group in np.unique(groups):
        train=groups!=group;test=groups==group
        scaler=StandardScaler().fit(data.X[train]);Z=scaler.transform(data.X[train]);Zt=scaler.transform(data.X[test])
        y=data.Y[train];ym=y.mean(axis=0);T=y-ym
        ols_beta=np.linalg.lstsq(np.column_stack([np.ones(len(Z)),Z]),y,rcond=None)[0]
        ridge_beta=np.linalg.solve(Z.T@Z+np.eye(Z.shape[1]),Z.T@T)
        independent={'OLS':np.column_stack([np.ones(len(Zt)),Zt])@ols_beta,
                     'Ridge':Zt@ridge_beta+ym}
        for model,mu in independent.items():
            for k,i in enumerate(np.flatnonzero(test)):
                p=pred[(pred.model==model)&(pred.experiment_id==data.ids[i])].set_index('target').loc[data.target_names]
                difference=float(np.max(abs(p.y_pred.to_numpy()-mu[k])))
                max_linear_error=max(max_linear_error,difference)
    assert max_linear_error<1e-9,max_linear_error
    legacy_max=None
    if name in legacy:
        old=pd.read_csv(ROOT/'outputs/results'/legacy[name]/'basic_predictions.csv')
        old=old[old.model=='GPR'].set_index(['experiment_id','target'])
        current=pred[pred.model=='ARD_Matern52'].set_index(['experiment_id','target']).loc[old.index]
        legacy_max=float(np.max(abs(current.y_pred-old.y_pred)))
        assert legacy_max<1e-8,(name,legacy_max)
    records.append({'scenario':name,'status':'passed','held_out_predictions':len(pred),
                    'original_ARD_GP_prediction_max_error':legacy_max,
                    'independent_OLS_Ridge_prediction_max_error':max_linear_error,
                    'checks':['source and implementation hashes','truths reconstructed from declared snapshot/variant',
                              'exactly one held-out prediction per model/target/record','source-group separation',
                              'recomputed RMSE and R2','independent least-squares/ridge matrix solutions']})
    for metric in ['CD Loss','AR','simulated response']:
        part=metrics[(metrics.metric==metric)&(metrics.structure!='all_mean')]
        for model,d in part.groupby('model'):
            summaries.append({'scenario':name,'model':model,'metric':metric,'target_count':len(d),
                              'r2_min':d.r2.min(),'r2_max':d.r2.max(),'mean_target_rmse':d.rmse.mean(),
                              'r2_positive_count':int((d.r2>0).sum())})
(OUT/'validation.json').write_text(json.dumps({'status':'passed','runs':records},indent=2),encoding='utf-8')
pd.DataFrame(summaries).to_csv(OUT/'structural_summary.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'validated_scenarios':len(records),'all_checks_passed':True}))
