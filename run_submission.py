"""Verify saved research or explicitly repeat the two 12-response protocols.

Portable reproduction requires identical rows and queries plus each test-AUC
relative error <= 3%; curve differences are reported separately. This is a
numerical acceptance rule, not a scientific equivalence or significance test.
Strict numerical mode retains an absolute 1e-8 check for the saved metrics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs/revision_v4/validation"
DEFAULT_AUC_RTOL=.03
RELATIVE_DENOMINATOR_EPSILON=1e-12
STRICT_ATOL=1e-8


class ReproductionComparisonError(ValueError):
    """Preserve all per-file diagnostics when a comparison does not pass."""
    def __init__(self, checks):
        self.checks=checks
        failed=[c["file"] for c in checks if c["status"]!="passed"]
        super().__init__("Reproduction comparison failed: "+", ".join(failed))


def _numeric_summary(reference, current, columns):
    a=reference[columns].to_numpy(float);b=current[columns].to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite values in required numerical metrics")
    absolute=np.abs(b-a)
    relative=absolute/np.maximum(np.abs(a),RELATIVE_DENOMINATOR_EPSILON)
    return {"max_numeric_difference":float(absolute.max()),
            "max_relative_difference":float(relative.max()),
            "relative_denominator":"max(abs(reference), epsilon)",
            "relative_denominator_epsilon":RELATIVE_DENOMINATOR_EPSILON,
            "columns":{column:{"max_absolute_difference":float(absolute[:,i].max()),
                               "max_relative_difference":float(relative[:,i].max()),
                               "p95_relative_difference":float(np.quantile(relative[:,i],.95)),
                               "p99_relative_difference":float(np.quantile(relative[:,i],.99)),
                               "rows_above_strict_atol":int((absolute[:,i]>STRICT_ATOL).sum())}
                       for i,column in enumerate(columns)}}


def compare_runs(original, reproduced, seed=None, *, mode="portable", auc_rtol=DEFAULT_AUC_RTOL):
    """Return diagnostics or raise with them; never silently change result CSVs.

    Portable mode checks both AUC columns where present. Query identity and
    structural columns stay exact in both modes. Strict means numerical metric
    agreement within an absolute tolerance, not byte-identical files.
    """
    if mode not in {"portable","strict"}:raise ValueError("Unknown comparison mode")
    if not np.isfinite(auc_rtol) or not 0<=auc_rtol<1:raise ValueError("auc_rtol must be finite and in [0,1)")
    checks=[]
    for name,keys,columns in [
        ("al_curves.csv",["seed","strategy","target","step"],["rmse","mae","nrmse","coverage95","mean_interval_width"]),
        ("al_auc.csv",["seed","strategy","target"],["test_auc","normalized_test_auc","mean_curve_rmse","final_rmse","final_nrmse"]),
        ("al_selections.csv",["seed","strategy","acquisition_step"],["experiment_id"]),
    ]:
        check={"file":name,"status":"failed","mode":mode}
        try:
            a=pd.read_csv(ROOT/"outputs/results"/original/name)
            b=pd.read_csv(ROOT/"outputs/results"/reproduced/name)
            if seed is not None:
                a=a[a.seed==seed];b=b[b.seed==seed]
            if a.empty or b.empty:raise ValueError("Empty comparison")
            if set(a.columns)!=set(b.columns):raise ValueError("Different result schema")
            # Older minimal CSVs may omit secondary metrics, but the principal
            # two curve metrics and test_auc are always required.
            required=keys+(["rmse","nrmse"] if name=="al_curves.csv" else ["test_auc"] if name=="al_auc.csv" else ["experiment_id"])
            if not set(required)<=set(a.columns):raise ValueError("Required columns missing")
            if a[keys].isna().any().any() or b[keys].isna().any().any():raise ValueError("Missing row keys")
            if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError("Duplicate row keys")
            a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
            if not a[keys].equals(b[keys]):raise ValueError("Different row identities")
            structural=[c for c in ["structure","metric","n_train","n_train_before","remaining_candidates_before"] if c in a]
            if structural and not a[structural].equals(b[structural]):raise ValueError("Different structural values or training sizes")
            columns=[c for c in columns if c in a]
            check.update(rows=len(a),row_identities="identical",structural_columns="identical",
                         **_numeric_summary(a,b,columns))
            if name=="al_selections.csv":
                mismatch=int((a.experiment_id!=b.experiment_id).sum())
                check.update(query_identities="identical" if not mismatch else "different",query_mismatches=mismatch)
                if mismatch:raise ValueError("Selected experiment IDs differ")
            elif mode=="strict":
                check["strict_absolute_tolerance"]=STRICT_ATOL
                if check["max_numeric_difference"]>STRICT_ATOL:raise ValueError("Strict numerical tolerance exceeded")
            elif name=="al_auc.csv":
                auc_columns=[c for c in ["test_auc","normalized_test_auc"] if c in a]
                integral=_numeric_summary(a,b,auc_columns)
                check.update(auc_relative_tolerance=auc_rtol,auc_max_relative_difference=integral["max_relative_difference"],
                             accepted_auc_columns=auc_columns)
                if integral["max_relative_difference"]>auc_rtol+8*np.finfo(float).eps:raise ValueError("Per-row AUC relative tolerance exceeded")
            else:check["curve_policy"]="Finite metric diagnostics only; portable acceptance is based on exact queries and per-row AUC"
            check["status"]="passed"
        except (OSError,ValueError,KeyError,TypeError) as error:
            check["error"]=str(error)
        checks.append(check)
    if any(c["status"]!="passed" for c in checks):raise ReproductionComparisonError(checks)
    return checks


def paired_macro_diagnostics(original,reproduced):
    """Report changes in paired effect sizes/signs without declaring equivalence."""
    frames=[pd.read_csv(ROOT/"outputs/results"/name/"al_auc.csv") for name in [original,reproduced]]
    result=[]
    for target in ["__macro__","__macro_cd__","__macro_ar__"]:
        for other in ["random","uncertainty","maximin"]:
            deltas=[]
            for frame in frames:
                part=frame[frame.target==target].pivot(index="seed",columns="strategy",values="test_auc").sort_index()
                deltas.append(part.duw-part[other])
            if not deltas[0].index.equals(deltas[1].index):raise ValueError("Different seeds in paired macro diagnostics")
            describe=lambda d:{"mean_delta_auc":float(d.mean()),"median_delta_auc":float(d.median()),
                               "wins":int((d < -1e-10).sum()),"losses":int((d > 1e-10).sum()),
                               "ties":int((abs(d)<=1e-10).sum())}
            reference,current=map(describe,deltas)
            result.append({"target":target,"comparison":"duw_minus_"+other,"reference":reference,"reproduced":current,
                           "mean_delta_change":current["mean_delta_auc"]-reference["mean_delta_auc"],
                           "win_loss_counts_unchanged":all(reference[k]==current[k] for k in ["wins","losses","ties"]),
                           "scope":"Diagnostic only: passing numerical tolerance does not establish scientific equivalence"})
    return result


def completed_run_matches(source,target):
    """Only reuse a complete, source/config/code-matched computation."""
    folder=ROOT/"outputs/results"/target
    meta=json.loads((folder/"run_metadata.json").read_text(encoding="utf-8"))
    reference=json.loads((ROOT/"outputs/results"/source/"run_metadata.json").read_text(encoding="utf-8"))
    if meta.get("status")!="complete" or set(meta.get("stages_completed",[]))!={"basic","al","optimization"}:
        raise ValueError("Existing target is not a complete computation")
    if meta["configuration"]!=reference["configuration"] or meta["source_sha256"]!=reference["source_sha256"]:
        raise ValueError("Existing target has different source/configuration")
    expected_configuration=json.loads((ROOT/"configs/real_main.json").read_text(encoding="utf-8-sig"))
    expected_configuration.update(exclude_ids=reference["configuration"]["exclude_ids"],cfd_variant="excel")
    if meta["configuration"]!=expected_configuration:raise ValueError("Current experiment configuration differs from completed target")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(ROOT/"data/raw/user_confirmed_table_2026-11-09.json")!=meta["source_sha256"]:
        raise ValueError("Current source differs from completed target")
    expected={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/"code/research").glob("*.py")}
    expected["run_research.py"]=sha(ROOT/"run_research.py")
    actual={rel.replace("\\","/"):value for rel,value in meta["code_sha256"].items()}
    if actual!=expected:raise ValueError("Existing target code hashes differ")
    for name in ["al_curves.csv","al_auc.csv","al_selections.csv","basic_predictions.csv"]:
        if not (folder/name).is_file():raise FileNotFoundError("Completed target missing "+name)
    return {"status":"reused_verified_complete","versions":meta.get("versions",{})}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-original",action="store_true")
    parser.add_argument("--run-prefix",default="reproduced",help="Fresh output-name prefix; saved paper results are preserved")
    parser.add_argument("--workers",type=int,choices=range(1,5),default=4)
    parser.add_argument("--comparison-mode",choices=["portable","strict"],default="portable",
                        help="portable: exact queries and per-row AUC relative tolerance; strict: absolute metric agreement")
    parser.add_argument("--auc-rtol",type=float,default=DEFAULT_AUC_RTOL,
                        help="Portable per-row AUC relative tolerance, default 0.03; not an equivalence margin")
    parser.add_argument("--reuse-completed",action="store_true",help="With --recompute-original, reuse complete source/config/code-matched prefix runs")
    parser.add_argument("--documents",action="store_true",help="Build DOCX/Markdown only; no PDF/PPT export")
    args=parser.parse_args()
    if Path(args.run_prefix).name!=args.run_prefix or args.run_prefix in {".",".."}:parser.error("Use a simple output prefix")
    if not np.isfinite(args.auc_rtol) or not 0<=args.auc_rtol<1:parser.error("--auc-rtol must be finite and in [0,1)")
    if args.reuse_completed and not args.recompute_original:parser.error("--reuse-completed requires --recompute-original")
    env=os.environ.copy();tmp=ROOT/"_tmp";tmp.mkdir(exist_ok=True)
    env.update(PYTHONUTF8="1",PYTHONIOENCODING="utf-8",PYTHONDONTWRITEBYTECODE="1",TEMP=str(tmp),TMP=str(tmp),TMPDIR=str(tmp))
    record={"status":"running","commands":[],"new_physical_experiments":False,
            "environment":{"python":platform.python_version(),"platform":platform.platform()},
            "comparison_policy":{"mode":args.comparison_mode,"query_ids":"exact","auc_relative_tolerance":args.auc_rtol,
                                 "relative_denominator_epsilon":RELATIVE_DENOMINATOR_EPSILON,"strict_absolute_tolerance":STRICT_ATOL,
                                 "scope":"Numerical reproduction acceptance only; not a scientific equivalence test or bytewise-reproduction claim"}}
    def run(script,extra=()):
        command=[sys.executable,"-B",script,*extra];print("RUN "+script,flush=True)
        started=time.perf_counter();result=subprocess.run(command,cwd=ROOT,env=env)
        record["commands"].append({"script":script,"arguments":list(extra),"exit_code":result.returncode,"elapsed_seconds":time.perf_counter()-started})
        if result.returncode:raise RuntimeError("Failed: "+script)
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        if args.recompute_original:
            record["original_protocol_reproduction"]=[]
            for source,exclude in [("real_main",[]),("real_drop24",["24"])]:
                target=args.run_prefix+"_"+source
                item={"source":source,"reproduced":target,"status":"running"}
                try:
                    if (ROOT/"outputs/results"/target).exists():
                        if not args.reuse_completed:raise FileExistsError("Choose a fresh --run-prefix or use --reuse-completed: "+target)
                        item["execution"]=completed_run_matches(source,target)
                    else:
                        extra=["--config","configs/real_main.json","--run-name",target,"--workers",str(args.workers)]
                        if exclude:extra += ["--exclude-ids",*exclude]
                        run("run_research.py",extra)
                        item["execution"]={"status":"computed"}
                    item["checks"]=compare_runs(source,target,mode=args.comparison_mode,auc_rtol=args.auc_rtol)
                    item["paired_macro_diagnostics"]=paired_macro_diagnostics(source,target)
                    item["status"]="passed"
                except Exception as error:
                    item.update(status="failed",error=str(error))
                    if isinstance(error,ReproductionComparisonError):
                        item["checks"]=error.checks
                        try:item["paired_macro_diagnostics"]=paired_macro_diagnostics(source,target)
                        except Exception as diagnostic_error:item["diagnostic_error"]=str(diagnostic_error)
                record["original_protocol_reproduction"].append(item)
                # Persist both successful and failed comparisons as each group finishes.
                (OUT/"submission_validation.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
            failures=[x["source"] for x in record["original_protocol_reproduction"] if x["status"]!="passed"]
            if failures:raise RuntimeError("Reproduction failed after checking both data scenarios: "+", ".join(failures))
        run("run_revision.py")
        run("code/revision/review_v3.py")
        run("-m",["pytest","tests/test_research_integrity.py","tests/test_reproduction_comparison.py","-q"])
        if args.documents:run("code/revision/build_documents_v4.py")
        record["status"]="passed"
    except Exception as error:
        record.update(status="failed",error=str(error))
        raise
    finally:
        record["entry_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (OUT/"submission_validation.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":record["status"]}))


if __name__=="__main__":main()
