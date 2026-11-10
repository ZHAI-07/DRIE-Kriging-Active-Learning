"""Regenerate research figures only from the authoritative table and saved runs.

All plotting uses matplotlib. SD bands describe repeated offline partitions,
not independent physical experiments. No simulation labels replace measurements.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_tmp" / "mplconfig"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "tools"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
COLORS = {"random": "#737373", "uncertainty": "#39779C",
          "maximin": "#C38A3F", "duw": "#4B725D"}
LABELS = {"random": "Random", "uncertainty": "Uncertainty",
          "maximin": "Maximin", "duw": "DUW-ALK"}
ORDER = list(COLORS)
FIGURES = ROOT / "outputs" / "figures"
QA = ROOT / "outputs" / "reports" / "figure_qa"
MANIFEST = []


def save(fig, name, question, sources, note, axes=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=axes, json_out=QA / f"{name}.alignment.json",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.svg")
    MANIFEST.append({"name": name, "question": question,
                     "source_files": sources, "note": note,
                     "png": f"outputs/figures/{name}.png",
                     "pdf": f"outputs/figures/{name}.pdf",
                     "backend": "Python matplotlib"})
    plt.close(fig)


def raw_figures():
    path = ROOT / "data/raw/user_confirmed_table_2026-11-09.json"
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    a = np.asarray(raw["rows"], float)
    structures = [s["label"] for s in raw["structures"]]
    for offset, metric, title, unit, cmap in (
        (5, "cd", "CD loss across experiments and structures", "CD loss (μm)", "cividis"),
        (6, "ar", "AR across experiments and structures", "Nominal AR", "viridis")):
        values = a[:, offset::2].T
        fig, ax = plt.subplots(figsize=(7.2, 2.7))
        fig.subplots_adjust(left=.09, right=.86, bottom=.22, top=.84)
        im = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set(xticks=np.arange(24), xticklabels=a[:, 0].astype(int),
               yticks=np.arange(6), yticklabels=structures, xlabel="Experiment ID",
               ylabel="Opening - line", title=title)
        for tick in ax.get_xticklabels():
            if tick.get_text() in {"2", "13", "24"}:
                tick.set_color("#A34D3C")
                tick.set_fontweight("bold")
        cax = fig.add_axes([.89, .22, .018, .62])
        fig.colorbar(im, cax=cax).set_label(unit)
        save(fig, f"01_{metric}_data_matrix", "How do the observed responses vary by process experiment and geometry?",
             [str(path.relative_to(ROOT))], "All 24 experiments and six structures retained. Highlighted IDs are audit flags, not exclusions.", [ax])
    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    fig.subplots_adjust(left=.10, right=.97, bottom=.23, top=.88)
    X = a[:, 1:5]
    z = (X - X.min(axis=0)) / np.ptp(X, axis=0)
    for i, row in enumerate(z):
        flagged = a[i, 0] == 24
        ax.plot(range(4), row, color="#A34D3C" if flagged else "#42738C",
                alpha=1.0 if flagged else .25, linewidth=1.8 if flagged else .7,
                marker="o", markersize=3, label="Experiment 24" if flagged else None)
    ax.set(xticks=range(4), xticklabels=["Coil\n840-1000 W", "Platen\n35-45 W", "SF6\n115-155 sccm", "Passivation\n8-12 mTorr"],
           ylabel="Position within observed range", ylim=(-.08, 1.12), title="Observed four-parameter design")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "02_process_design", "Which process settings are represented by the 24 experiments?",
         [str(path.relative_to(ROOT))], "Per-feature min-max normalization is descriptive visualization only; it is not used as a fitted model transform.")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.90, wspace=.36, hspace=.58)
    for s, ax in enumerate(axes.flat):
        cd, ar = a[:, 5+2*s], a[:, 6+2*s]
        ax.scatter(cd, ar, s=20, c="#42738C", alpha=.8, edgecolors="white", linewidths=.3)
        ax.scatter(cd[-1], ar[-1], s=48, marker="s", facecolor="none", edgecolor="#A34D3C", linewidth=1.1)
        ax.set(title=structures[s], xlabel="CD loss (μm)", ylabel="Nominal AR")
    save(fig, "03_observed_cd_ar", "What trade-offs are present in the measured CD loss and AR?",
         [str(path.relative_to(ROOT))], "24 observations per structure. Square marks experiment 24; all points are measured-table values, not optimized predictions.", list(axes.flat))


def basic_figures(run):
    folder = ROOT / "outputs/results" / run
    path = folder / "basic_predictions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    models = [m for m in df.model.unique() if "gpr" in m.lower() or "kriging" in m.lower()]
    if not models:
        raise ValueError(f"No GPR predictions in {path}")
    preferred = next((m for m in models if "matern" in m.lower()), models[0])
    df = df[df.model == preferred]
    for metric in df.metric.unique():
        chunk = df[df.metric == metric]
        structures = list(chunk.structure.unique())
        multi = len(structures) == 6
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0)) if multi else plt.subplots(figsize=(4.3, 3.7))
        axes = np.atleast_1d(axes).ravel()
        fig.subplots_adjust(left=.11, right=.98, bottom=.12, top=.89, wspace=.40, hspace=.60) if multi else fig.subplots_adjust(left=.17,right=.97,bottom=.17,top=.88)
        unit = " (μm)" if metric == "CD Loss" else ""
        for ax, structure in zip(axes, structures):
            d = chunk[chunk.structure == structure]
            lo = min(d.y_true.min(), d.y_pred.min())
            hi = max(d.y_true.max(), d.y_pred.max())
            pad = max((hi-lo)*.10, .01)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle="--", color="#777777", linewidth=.8)
            ax.scatter(d.y_true, d.y_pred, color="#42738C", s=21, alpha=.85, edgecolors="white", linewidths=.3)
            flagged = d[d.experiment_id == 24] if run.startswith("real") else d.iloc[:0]
            ax.scatter(flagged.y_true, flagged.y_pred, s=52, marker="s", facecolors="none", edgecolors="#A34D3C")
            ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), title=str(structure),
                   xlabel="Observed"+unit, ylabel="Held-out prediction"+unit)
        slug = "cd" if metric == "CD Loss" else "ar" if metric == "AR" else "cfd"
        save(fig, f"04_{run}_{slug}_prediction", f"Does GPR predict held-out {metric} within each structure?",
             [str(path.relative_to(ROOT))], "Out-of-source-group predictions. IDs 2 and 13 are held out together. Dashed line is identity, not a fit. No training-fit claims.", list(axes))
        # Every residual is shown, including flagged observations.
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        fig.subplots_adjust(left=.11,right=.98,bottom=.19,top=.85)
        for s, structure in enumerate(structures):
            d = chunk[chunk.structure == structure].sort_values("experiment_id")
            ax.plot(d.experiment_id, d.y_pred-d.y_true, marker="o", markersize=3, linewidth=.8, alpha=.8, label=str(structure))
        ax.axhline(0, color="#777777", linewidth=.7, linestyle="--")
        ax.set(xlabel="Simulation record ID" if run.startswith("cfd") else "Experiment ID", ylabel="Prediction - observation"+unit, title=f"{metric}: held-out residuals")
        ax.legend(ncol=min(6,len(structures)),loc="upper center",bbox_to_anchor=(.5,1.23),frameon=False)
        save(fig, f"05_{run}_{slug}_residual", f"Which held-out experiments dominate {metric} prediction error?",
             [str(path.relative_to(ROOT))], "All held-out residuals retained; curves connect experiment IDs only and do not imply temporal ordering.")


def al_figures(run):
    folder = ROOT / "outputs/results" / run
    path = folder / "al_curves.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    targets = [t for t in ["__macro__", "__macro_cd__", "__macro_ar__"] if t in set(df.target)]
    for target in targets:
        d = df[df.target == target]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        fig.subplots_adjust(left=.14, right=.98, bottom=.17, top=.82)
        metric = "nrmse" if target == "__macro__" else "rmse"
        for method in ORDER:
            part = d[d.strategy == method]
            if part.empty:
                continue
            stats = part.groupby("n_train")[metric].agg(["mean", "std"]).sort_index()
            x, y, sd = stats.index.to_numpy(), stats["mean"].to_numpy(), stats["std"].fillna(0).to_numpy()
            ax.plot(x, y, color=COLORS[method], linewidth=2.0 if method=="duw" else 1.4,
                    marker="o", markersize=3.5, label=LABELS[method])
            ax.fill_between(x, np.maximum(y-sd,0), y+sd, color=COLORS[method], alpha=.10, linewidth=0)
        ax.set(xlabel="Labeled simulation records" if run.startswith("cfd") else "Labeled process experiments", ylabel="Mean normalized RMSE" if target=="__macro__" else "Mean CD loss RMSE (μm)" if target=="__macro_cd__" else "Mean AR RMSE")
        ax.set_xticks(sorted(d.n_train.unique()))
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5,1.23), frameon=False, columnspacing=1)
        repeats=d.seed.nunique()
        slug=target.strip("_")
        save(fig, f"06_{run}_{slug}_learning", "Does active selection improve held-out prediction at equal label budgets?",
             [str(path.relative_to(ROOT))], f"Mean +/- one SD over {repeats} paired offline repetitions on a fixed dataset. Bands are not confidence intervals or independent physical replication. Initial-training output scales are fixed over each learning curve.")
    auc_path=folder/"al_auc.csv"
    auc=pd.read_csv(auc_path)
    d=auc[auc.target=="__macro__"].pivot(index="seed",columns="strategy",values="test_auc")
    methods=[m for m in ORDER if m!="random" and m in d]
    if methods:
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        fig.subplots_adjust(left=.17,right=.97,bottom=.18,top=.86)
        for pos,method in enumerate(methods):
            delta=(d[method]-d["random"]).to_numpy()
            # Horizontal offsets separate displayed seeds; measured deltas are unchanged.
            ax.scatter(pos+np.linspace(-.10,.10,len(delta)),delta,s=21,alpha=.65,c=COLORS[method],edgecolor="white",linewidth=.3)
            q=np.quantile(delta,[.25,.5,.75])
            ax.plot([pos+.18,pos+.18],[q[0],q[2]],color=COLORS[method],linewidth=3)
            ax.plot([pos+.12,pos+.24],[q[1],q[1]],color="black",linewidth=1.4)
        ax.axhline(0,color="#777777",linestyle="--",linewidth=.8)
        ax.set(xticks=range(len(methods)),xticklabels=[LABELS[m] for m in methods],ylabel="Paired TestAUC - Random",title="Lower than zero favors active selection")
        save(fig,f"07_{run}_paired_auc","Are improvements over random consistent across matched partitions?",
             [str(auc_path.relative_to(ROOT))],"Each dot is one paired random-partition comparison. Vertical segments mark IQR and horizontal marks medians. TestAUC integrates prediction error over acquisition steps, not ROC area.")


def synthesis_figures():
    basic_path=ROOT/"outputs/results/real_main/basic_metrics.csv"
    if basic_path.exists():
        df=pd.read_csv(basic_path)
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.5))
        fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.79,wspace=.40)
        for ax,metric in zip(axes,["CD Loss","AR"]):
            d=df[df.metric==metric]
            structures=list(d.structure.unique())
            for model,color,marker in [("Mean","#737373","o"),("Ridge","#C38A3F","s"),("GPR","#39779C","^")]:
                part=d[d.model==model].set_index("structure").loc[structures]
                ax.plot(range(6),part.rmse,marker=marker,markersize=4,color=color,label=model,linewidth=1)
            ax.set(xticks=range(6),xticklabels=structures,xlabel="Structure",title=metric,
                   ylabel="Held-out RMSE (μm)" if metric=="CD Loss" else "Held-out RMSE")
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.54,1),frameon=False)
        save(fig,"08_basic_model_comparison","Does GPR improve over simpler prediction baselines?",[str(basic_path.relative_to(ROOT))],
             "Same 23 source-group hold-outs for all models; 24 pooled predictions per response. Connected structures are categorical, not a continuous size response.",list(axes))
    runs=[("real_main","All 24"),("real_drop2","Omit 2"),("real_drop13","Omit 13"),("real_drop24","Omit 24"),("real_historical_schedule","Historical schedule")]
    if all((ROOT/"outputs/results"/run/"al_auc.csv").exists() for run,_ in runs):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.6))
        fig.subplots_adjust(left=.23,right=.98,bottom=.22,top=.88,wspace=.32)
        sources=[]
        records=[]
        for ax,target,title in zip(axes,["__macro_cd__","__macro_ar__"],["CD loss","AR"]):
            for position,(run,label) in enumerate(runs):
                path=ROOT/"outputs/results"/run/"al_auc.csv"
                d=pd.read_csv(path)
                p=d[d.target==target].pivot(index="seed",columns="strategy",values="test_auc")
                values=100*(p["random"]-p["duw"])/p["random"]
                rng=np.random.default_rng(20261109)
                boot=values.to_numpy()[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
                low,high=np.quantile(boot,[.025,.975]); mean=float(values.mean())
                ax.plot([low,high],[position,position],color="#4B725D",linewidth=1.3)
                ax.scatter([mean],[position],color="#4B725D",s=24,zorder=3)
                records.append({"run":run,"target":target,"n_seeds":len(values),"mean_relative_improvement_percent":mean,"conditional_bootstrap_ci_low":float(low),"conditional_bootstrap_ci_high":float(high)})
                sources.append(str(path.relative_to(ROOT)))
            ax.axvline(0,color="#888888",linestyle="--",linewidth=.8)
            ax.set(yticks=range(len(runs)),yticklabels=[x[1] for x in runs] if ax is axes[0] else [],
                   ylim=(len(runs)-.5,-.5),xlabel="Mean relative improvement (%)",title=title)
        save(fig,"09_duw_sensitivity","Does DUW improvement over Random persist under the source and schedule sensitivities?",list(dict.fromkeys(sources)),
             "Mean of paired per-seed percentages; positive favors DUW. Bars are percentile bootstrap intervals over 30 seeds on each fixed dataset, not physical-population intervals. Source exclusion changes the task; it is not evidence justifying deletion.",list(axes))
        (ROOT/"outputs/reports/sensitivity_figure_values.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    status_path=ROOT/"outputs/results/real_main/optimization_status.json"
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding="utf-8"))
        path=ROOT/"data/raw/user_confirmed_table_2026-11-09.json"
        raw=json.loads(path.read_text(encoding="utf-8-sig")); a=np.asarray(raw["rows"],float)
        ids=status["observed_pareto_ids"]
        selected=np.isin(a[:,0],ids); Y=a[:,5:]
        z=(Y-Y.min(axis=0))/np.ptp(Y,axis=0); z[:,::2]=1-z[:,::2]
        fig,ax=plt.subplots(figsize=(7.2,3.9))
        fig.subplots_adjust(left=.13,right=.85,bottom=.23,top=.86)
        im=ax.imshow(z[selected],aspect="auto",cmap="cividis",vmin=0,vmax=1)
        ax.set(yticks=range(len(ids)),yticklabels=ids,ylabel="Measured experiment ID",xticks=range(12),
               xticklabels=[f"{s['label']}\n{m}" for s in raw["structures"] for m in ["CD","AR"]],
               title="Non-dominated alternatives in the measured table")
        cax=fig.add_axes([.89,.23,.018,.63]);fig.colorbar(im,cax=cax).set_label("Relative benefit within observed range")
        save(fig,"10_measured_pareto","Which measured recipes are not dominated across all 12 objectives?",
             [str(path.relative_to(ROOT)),str(status_path.relative_to(ROOT))],
             "Minimize six CD losses and maximize six ARs without weights. Display scales only: CD=1-minmax, AR=minmax over all 24 rows. These are finite measured alternatives, not validated optima; experiment 24 retains its unresolved provenance flag.",[ax])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",nargs="*",default=["real_main","cfd_main"])
    parser.add_argument("--data-only",action="store_true")
    parser.add_argument("--synthesis-only",action="store_true")
    args=parser.parse_args()
    if not args.synthesis_only:
        raw_figures()
    if not args.data_only and not args.synthesis_only:
        for run in args.runs:
            basic_figures(run)
            al_figures(run)
    if not args.data_only:
        synthesis_figures()
    manifest_path=ROOT/"outputs/reports/figure_manifest.json"
    old=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    combined={item["name"]:item for item in old+MANIFEST}
    manifest_path.write_text(json.dumps(list(combined.values()),ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"figures_generated":len(MANIFEST),"names":[x["name"] for x in MANIFEST]}))


if __name__=="__main__":
    main()
