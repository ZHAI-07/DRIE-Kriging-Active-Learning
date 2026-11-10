"""Review-driven model comparison; no v1 sources or results are overwritten.

Universal Kriging uses GLS linear coefficients, REML covariance fitting, and
the additional trend-estimation predictive variance from GPML eq. 2.42.
All reported predictions are held out by source group. No active-learning
method is selected automatically from these post-review comparisons.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, check_grad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from research.core import IndependentGP, GPConfig, safe_scale
from research.data import load_dataset, source_groups, make_split
from research.experiments import clean_json, frame_csv, write_json

OUT = ROOT / "outputs/revision/models"
SCENARIOS = {
    "real_all24": ("real", [], "excel"),
    "real_without24": ("real", [24], "excel"),
    "real_without2_24": ("real", [2, 24], "excel"),
    "real_without13_24": ("real", [13, 24], "excel"),
    "cfd_excel": ("cfd", [], "excel"),
    "cfd_legacy": ("cfd", [], "legacy_code"),
}
CONFIG = GPConfig()


def covariance(ard=False, dimensions=4):
    return (ConstantKernel(1., (1e-3, 1e3)) *
            Matern(np.ones(dimensions) if ard else 1., (1e-2, 10.), nu=2.5) +
            WhiteKernel(1e-3, (1e-6, .1)))


class IsotropicGP(IndependentGP):
    """Reusable raw-X/raw-Y interface for independently specified v2 AL checks."""
    def fit(self, X, Y, x_scaler=None):
        X,Y=np.asarray(X,float),np.asarray(Y,float)
        if Y.ndim==1:Y=Y[:,None]
        self.x_scaler=x_scaler if x_scaler is not None else StandardScaler().fit(X)
        self.y_mean,self.y_scale=Y.mean(axis=0),safe_scale(Y)
        Z=self.x_scaler.transform(X);T=(Y-self.y_mean)/self.y_scale
        self.models,self.fit_warnings,self.optimizer_runs=[],[],[]
        for j in range(Y.shape[1]):
            kernel=ConstantKernel(1.,self.config.constant_bounds)*Matern(1.,self.config.length_bounds,nu=2.5)
            kernel+=WhiteKernel(self.config.noise_initial,self.config.noise_bounds)
            gp=GaussianProcessRegressor(kernel=kernel,alpha=self.config.alpha,normalize_y=False,
                                        optimizer=self._optimizer if self.config.optimize else None,
                                        n_restarts_optimizer=self.config.restarts,random_state=self.seed+j)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always");gp.fit(Z,T[:,j])
            self.fit_warnings.extend({"target_index":j,"message":str(w.message)} for w in caught)
            self.models.append(gp)
        return self


def append_mean_endpoints(data):
    """Fixed equal-weight exploratory endpoints; not mean of structural errors."""
    if data.name!="real":return data
    data.Y=np.column_stack([data.Y,data.Y[:,::2].mean(axis=1),data.Y[:,1::2].mean(axis=1)])
    data.target_names=list(data.target_names)+["mean_CD","mean_AR"]
    data.structures=list(data.structures)+["all_mean","all_mean"]
    data.metrics=list(data.metrics)+["CD Loss","AR"]
    return data


class UniversalKriging:
    """Scalar Gaussian observation model on already standardized training data.

    f(z)=h(z)^T beta+g(z); h=[1,z]. beta is estimated by GLS with diffuse
    coefficient prior. Three log covariance parameters are optimized using
    restricted likelihood; fixed alpha is additional train-only regularization.
    Returned observation variance includes WhiteKernel and trend uncertainty,
    but excludes hyperparameter estimation uncertainty and extra test alpha.
    """
    def __init__(self, alpha=1e-3, maxiter=60, optimize=True):
        self.alpha, self.maxiter, self.optimize = alpha, maxiter, optimize

    @staticmethod
    def basis(Z):
        return np.column_stack([np.ones(len(Z)), np.asarray(Z, float)])

    def terms(self, theta, gradient=False):
        kernel = self.kernel.clone_with_theta(theta)
        if gradient:
            K, dK = kernel(self.Z, eval_gradient=True)
        else:
            K = kernel(self.Z)
        K = K + np.eye(len(K)) * self.alpha
        chol = cho_factor(K, lower=True, check_finite=False)
        KiH = cho_solve(chol, self.H, check_finite=False)
        Kiy = cho_solve(chol, self.y, check_finite=False)
        G = self.H.T @ KiH
        chol_G = cho_factor(G, lower=True, check_finite=False)
        beta = cho_solve(chol_G, self.H.T @ Kiy, check_finite=False)
        residual = self.y-self.H @ beta
        precision_residual = cho_solve(chol, residual, check_finite=False)
        logdet_K = 2*np.log(np.diag(chol[0])).sum()
        logdet_G = 2*np.log(np.diag(chol_G[0])).sum()
        nll = .5*(residual @ precision_residual + logdet_K + logdet_G +
                   (len(self.y)-self.H.shape[1])*np.log(2*np.pi))
        if not gradient:
            return nll, (kernel, chol, chol_G, KiH, beta, precision_residual)
        Ki = cho_solve(chol, np.eye(len(K)), check_finite=False)
        P = Ki-KiH @ cho_solve(chol_G, KiH.T, check_finite=False)
        d_nll = .5*np.einsum("ij,ijk->k", P-np.outer(precision_residual, precision_residual), dK)
        return float(nll), d_nll

    def fit(self, Z, y, fixed_theta=None):
        self.Z, self.y = np.asarray(Z, float), np.asarray(y, float)
        self.H = self.basis(self.Z)
        self.rank = np.linalg.matrix_rank(self.H)
        if len(self.Z) <= self.H.shape[1] or self.rank != self.H.shape[1]:
            raise ValueError("Universal Kriging needs n>p and a full-rank trend design")
        self.kernel = covariance(dimensions=self.Z.shape[1])
        initial = self.kernel.theta if fixed_theta is None else np.asarray(fixed_theta, float)
        self.optimizer_record = None
        if self.optimize and fixed_theta is None:
            result = minimize(lambda t: self.terms(t, True), initial, jac=True,
                              method="L-BFGS-B", bounds=self.kernel.bounds,
                              options={"maxiter": self.maxiter, "ftol": 1e-9, "gtol": 1e-5})
            theta = result.x
            self.optimizer_record = {"success":bool(result.success),"status":int(result.status),
                                     "iterations":int(result.nit),"message":str(result.message)}
        else:
            theta = initial
        self.nll, state = self.terms(theta)
        self.kernel_, self.chol, self.chol_G, self.KiH, self.beta, self.precision_residual = state
        return self

    def predict(self, Z, components=False):
        Z = np.asarray(Z, float)
        h = self.basis(Z)
        cross = self.kernel_(self.Z, Z)  # WhiteKernel has zero train/test covariance.
        solved = cho_solve(self.chol, cross, check_finite=False)
        mu = h @ self.beta + cross.T @ self.precision_residual
        simple_var = self.kernel_.diag(Z)-np.einsum("ij,ij->j",cross,solved)
        discrepancy = h-cross.T @ self.KiH
        trend_var = np.einsum("ij,ji->i", discrepancy,
                              cho_solve(self.chol_G, discrepancy.T, check_finite=False))
        variance = simple_var+trend_var
        if np.min(variance)<-1e-8:
            raise ArithmeticError("Materially negative predictive variance")
        if components:
            return mu, simple_var, trend_var
        return mu, np.sqrt(np.maximum(variance,0.))

    def diagnostics(self):
        theta=self.kernel_.theta;bounds=self.kernel_.bounds
        return {"kernel":str(self.kernel_),"theta":theta.tolist(),"beta":self.beta.tolist(),
                "restricted_negative_log_likelihood":self.nll,"trend_rank":int(self.rank),
                "n_minus_p":len(self.y)-self.H.shape[1],
                "trend_information_condition":float(np.linalg.cond(self.H.T @ self.KiH)),
                "boundary_hyperparameters":[int(i) for i in range(len(theta))
                    if min(abs(theta[i]-bounds[i,0]),abs(theta[i]-bounds[i,1]))<1e-4],
                "optimizer":self.optimizer_record}


def fit_isotropic(Z, T, test_Z):
    pred=[];std=[];diag=[]
    for j in range(T.shape[1]):
        optimizer_runs=[]
        def optimizer(obj,theta,bounds):
            r=minimize(obj,theta,jac=True,method="L-BFGS-B",bounds=bounds,
                       options={"maxiter":60,"ftol":1e-9,"gtol":1e-5})
            optimizer_runs.append({"success":bool(r.success),"message":str(r.message),"iterations":int(r.nit)})
            return r.x,float(r.fun)
        gp=GaussianProcessRegressor(kernel=covariance(dimensions=Z.shape[1]),alpha=1e-3,
                                    optimizer=optimizer,normalize_y=False,random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gp.fit(Z,T[:,j])
        mu,sd=gp.predict(test_Z,return_std=True)
        pred.append(mu);std.append(sd)
        diag.append({"target_index":j,"kernel":str(gp.kernel_),"theta":gp.kernel_.theta.tolist(),
                     "warnings":[str(w.message) for w in caught],"optimizer_runs":optimizer_runs})
    return np.column_stack(pred),np.column_stack(std),diag


def evaluate_scenario(name):
    started=time.perf_counter()
    kind,excluded,variant=SCENARIOS[name]
    data=append_mean_endpoints(load_dataset(ROOT,kind,excluded,variant));groups=source_groups(data)
    records=[];diagnostics=[]
    for fold,group in enumerate(np.unique(groups)):
        train=np.flatnonzero(groups!=group);test=np.flatnonzero(groups==group)
        X,Y=data.X[train],data.Y[train]
        scaler=StandardScaler().fit(X);Z=scaler.transform(X);Zt=scaler.transform(data.X[test])
        ym,ys=Y.mean(axis=0),safe_scale(Y);T=(Y-ym)/ys
        preds={"Mean":(np.tile(ym,(len(test),1)),None)}
        for label,reg in [("OLS",LinearRegression()),("Ridge",Ridge(alpha=1.))]:
            reg.fit(Z,T)
            preds[label]=(np.asarray(reg.predict(Zt)).reshape(len(test),Y.shape[1])*ys+ym,None)
            diagnostics.append({"fold":fold,"model":label,"train_ids":data.ids[train].tolist(),
                                "test_ids":data.ids[test].tolist(),"standardized_coefficients":reg.coef_.tolist(),
                                "intercept":np.asarray(reg.intercept_).tolist(),"X_mean":scaler.mean_.tolist(),
                                "X_scale":scaler.scale_.tolist(),"Y_mean":ym.tolist(),"Y_scale":ys.tolist()})
        old=IndependentGP(CONFIG,seed=fold).fit(X,Y,x_scaler=scaler)
        preds["ARD_Matern52"]=old.predict(data.X[test])
        diagnostics.append({"fold":fold,"model":"ARD_Matern52","train_ids":data.ids[train].tolist(),
                            "test_ids":data.ids[test].tolist(),**old.diagnostics()})
        if kind=="real":
            mu,sd,detail=fit_isotropic(Z,T,Zt)
            preds["Isotropic_Matern52"]=(mu*ys+ym,sd*ys)
            diagnostics.append({"fold":fold,"model":"Isotropic_Matern52","train_ids":data.ids[train].tolist(),
                                "test_ids":data.ids[test].tolist(),"targets":detail})
            mu=[];sd=[];detail=[]
            for j in range(Y.shape[1]):
                model=UniversalKriging().fit(Z,T[:,j]);m,s=model.predict(Zt)
                mu.append(m);sd.append(s);detail.append({"target_index":j,**model.diagnostics()})
            preds["UniversalKriging_iso_REML"]=(np.column_stack(mu)*ys+ym,np.column_stack(sd)*ys)
            diagnostics.append({"fold":fold,"model":"UniversalKriging_iso_REML","train_ids":data.ids[train].tolist(),
                                "test_ids":data.ids[test].tolist(),"targets":detail})
        else:
            polynomial=PolynomialFeatures(degree=2,include_bias=False)
            P=polynomial.fit_transform(Z);Pt=polynomial.transform(Zt)
            poly_scaler=StandardScaler().fit(P)
            response_surface=Ridge(alpha=1.).fit(poly_scaler.transform(P),T[:,0])
            preds["Quadratic_Ridge"]=(response_surface.predict(poly_scaler.transform(Pt))[:,None]*ys+ym,None)
            svr=SVR(kernel="rbf",C=10.,epsilon=.1,gamma="scale").fit(Z,T[:,0])
            preds["SVR_fixed"]=(svr.predict(Zt)[:,None]*ys+ym,None)
            diagnostics.append({"fold":fold,"model":"CFD_nonlinear_baselines","train_ids":data.ids[train].tolist(),
                                "test_ids":data.ids[test].tolist(),"quadratic_features":polynomial.get_feature_names_out().tolist(),
                                "quadratic_coefficients":response_surface.coef_.tolist(),"SVR_gamma":float(svr._gamma),
                                "SVR_support_vector_count":len(svr.support_)})
        for model,(mu,sd) in preds.items():
            for k,idx in enumerate(test):
                for j,target in enumerate(data.target_names):
                    st=float(sd[k,j]) if sd is not None else np.nan
                    records.append({"scenario":name,"model":model,"fold":fold,"source_group":int(group),
                                    "experiment_id":int(data.ids[idx]),"n_train":len(train),
                                    "target":target,"structure":data.structures[j],"metric":data.metrics[j],
                                    "y_true":data.Y[idx,j],"y_pred":mu[k,j],"std":st,
                                    "lower95":mu[k,j]-1.96*st,"upper95":mu[k,j]+1.96*st})
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records);frame_csv(frame,folder/"predictions.csv")
    metrics=[]
    for key,part in frame.groupby(["model","target","structure","metric"],sort=False):
        model,target,structure,metric=key;error=part.y_pred-part.y_true
        metrics.append({"scenario":name,"model":model,"target":target,"structure":structure,"metric":metric,
                        "n":len(part),"rmse":np.sqrt(np.mean(error**2)),"mae":np.mean(abs(error)),
                        "r2":r2_score(part.y_true,part.y_pred),
                        "coverage95":np.mean((part.y_true>=part.lower95)&(part.y_true<=part.upper95)) if part['std'].notna().all() else np.nan})
    frame_csv(pd.DataFrame(metrics),folder/"metrics.csv")
    write_json(folder/"fit_diagnostics.json",clean_json(diagnostics))
    write_json(folder/"run_metadata.json",{
        "status":"complete","scenario":name,"dataset":kind,"excluded_ids":excluded,"cfd_variant":variant,
        "source":str(Path(data.source_path).relative_to(ROOT)),"source_sha256":data.source_sha256,
        "numerical_y_sha256":hashlib.sha256(data.Y.tobytes()).hexdigest(),"ids":data.ids.tolist(),
        "X_shape":list(data.X.shape),"Y_shape":list(data.Y.shape),"source_groups":len(np.unique(groups)),
        "models":list(frame.model.unique()),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds":time.perf_counter()-started,"hyperparameter_selection":"fixed policies declared in protocol.json, no outer score selection",
        "interval_scope":"Gaussian fixed-fitted-hyperparameter predictive intervals; UK includes trend coefficient uncertainty but not covariance hyperparameter uncertainty",
        "derived_endpoints":"mean_CD and mean_AR are arithmetic means of six same-metric observed responses, appended to 12 original targets. They are not a claim of equal engineering utility or independent replication.",
        "dataset_policy":"Named exclusion scenarios are exploratory sensitivity strata, not verified corrections to source data"})
    return name,metrics,time.perf_counter()-started


def write_protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"protocol.json",{
        "declared_before_execution":datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "purpose":"Post-review exploratory model audit; not a confirmatory generalization claim or automatic AL model selection",
        "scenarios":SCENARIOS,"original_gp_configuration":asdict(CONFIG),
        "new_gp":"isotropic Matérn 5/2, same amplitude/noise/length bounds, alpha and optimizer settings as original GP",
        "uk":"Linear trend [1,z1,z2,z3,z4], GLS beta, REML theta, full trend variance correction; no plug-in OLS residual shortcut",
        "scaling":"X and Y standardized on each outer training fold only; quadratic features additionally standardized using training fold only",
        "baselines":{"Ridge_alpha":1.,"Quadratic_Ridge_alpha":1.,"SVR":{"kernel":"rbf","C":10.,"epsilon":.1,"gamma":"scale"}},
        "derived_endpoints":{"mean_CD":"arithmetic mean of six CD Loss values, micrometres","mean_AR":"arithmetic mean of six nominal AR values, dimensionless","scope":"Fixed equal-weight exploratory endpoints after review; not RMSE macro average or engineering acceptance substitute"},
        "evaluation":"Same source-group leave-one-out folds for every model; source IDs 2/13 together if both retained",
        "small_n_limit":"Initial AL n=6 with p=5 unrestricted trend coefficients leaves only n-p=1 if full rank; no UK active-learning claim or automatic deployment",
        "primary_reference":{"authors":"Carl Edward Rasmussen and Christopher K. I. Williams","title":"Gaussian Processes for Machine Learning","year":2006,
                             "url":"https://gaussianprocess.org/gpml/chapters/RW2.pdf","basis":"Sections 2.7 and 2.7.1, equations 2.42 and 2.45, accessed 2026-11-09"},
        "inference":"Model-conditional GP likelihood noise is not an independently measured noise floor; without repeats it cannot separate process variability from measurement noise/model misspecification"})
    data=load_dataset(ROOT,"real");rows=[]
    for seed in range(20261109,20261139):
        split=make_split(data,seed,6,6,8);X=data.X[split['initial']]
        Z=StandardScaler().fit_transform(X);H=UniversalKriging.basis(Z)
        rows.append({"seed":seed,"n_initial":len(X),"trend_columns":H.shape[1],"trend_rank":int(np.linalg.matrix_rank(H)),
                     "n_minus_rank":len(X)-np.linalg.matrix_rank(H),"full_rank":bool(np.linalg.matrix_rank(H)==H.shape[1]),
                     "condition_H":float(np.linalg.cond(H))})
    frame_csv(pd.DataFrame(rows),OUT/"initial_trend_identifiability.csv")


def mathematical_tests():
    rng=np.random.default_rng(20261109);Z=rng.normal(size=(14,3));y=rng.normal(size=14);test=rng.normal(size=(3,3))
    model=UniversalKriging(optimize=False).fit(Z,y);theta=model.kernel_.theta
    grad_error=check_grad(lambda t:model.terms(t)[0],lambda t:model.terms(t,True)[1],theta)
    assert grad_error<2e-5,grad_error
    # An independently solved universal-kriging saddle-point system supplies the
    # unbiased prediction weights; this does not reuse our GLS prediction code.
    K=model.kernel_(Z)+np.eye(len(Z))*model.alpha;H=model.H;cross=model.kernel_(Z,test);h=model.basis(test)
    system=np.block([[K,H],[H.T,np.zeros((H.shape[1],H.shape[1]))]])
    solution=np.linalg.solve(system,np.vstack([cross,h.T]));weights=solution[:len(Z)]
    mu,sv,tv=model.predict(test,components=True)
    reference_var=model.kernel_.diag(test)-np.sum(weights*cross,axis=0)-np.sum(solution[len(Z):]*h.T,axis=0)
    assert np.allclose(mu,weights.T@y,atol=1e-9)
    assert np.allclose(sv+tv,reference_var,atol=1e-9)
    assert np.all(tv>=-1e-10)
    affine=np.array([.2,-.4,.6,.1]);shifted=UniversalKriging(optimize=False).fit(Z,y+H@affine,fixed_theta=theta)
    mu2,sd2=shifted.predict(test);mu1,sd1=model.predict(test)
    assert np.allclose(mu2-mu1,h@affine,atol=1e-9) and np.allclose(sd1,sd2,atol=1e-9)
    failed=False
    try:UniversalKriging().fit(np.ones((6,4)),np.arange(6.))
    except ValueError:failed=True
    assert failed
    write_json(OUT/"mathematical_validation.json",{"status":"passed","REML_analytic_gradient_finite_difference_error":grad_error,
        "checks":["Analytic REML gradient agrees with independent finite differences","Mean and observation variance agree with saddle-point universal-Kriging solve",
                  "Trend coefficient uncertainty correction is nonnegative","Affine trend equivariance at fixed covariance parameters",
                  "Rank-deficient initial trend design rejected instead of pseudo-inverse hiding nonidentifiability"]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4);parser.add_argument("--scenarios",nargs="*",default=list(SCENARIOS));parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args();write_protocol();mathematical_tests()
    if args.test_only:print("Universal-Kriging mathematical checks passed");return
    all_metrics=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(evaluate_scenario,name):name for name in args.scenarios}
        for future in as_completed(futures):
            name,metrics,elapsed=future.result();all_metrics.extend(metrics);print(f"COMPLETE {name} {elapsed:.1f}s",flush=True)
    frame_csv(pd.DataFrame(all_metrics),OUT/"all_metrics.csv")
    print("Model audit complete; no automatic best-model or AL selection.",flush=True)


if __name__=="__main__":
    main()
