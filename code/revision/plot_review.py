"""Plot the revised claim chain from saved evidence, not model reruns."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"_tmp/mplconfig"))
sys.path.insert(0,str(ROOT/"code/tools"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

FIG=ROOT/"outputs/revision/figures"
QA=FIG/"qa"
FIG.mkdir(parents=True,exist_ok=True);QA.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":9,"axes.titlesize":10,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
 "legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False,
 "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
MANIFEST=[]

def save(fig,name,axes,sources,caption):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=QA/f"{name}.alignment.json",tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(FIG/f"{name}.png",dpi=600)
    fig.savefig(FIG/f"{name}.pdf")
    fig.savefig(FIG/f"{name}.svg")
    MANIFEST.append({"name":name,"png":f"outputs/revision/figures/{name}.png","pdf":f"outputs/revision/figures/{name}.pdf",
                     "source_sha256":{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},"caption":caption})
    plt.close(fig)


def comparison_and_strata():
    pair_path="outputs/revision/evidence/duw_pairwise_per_seed.csv"
    split_path="outputs/results/real_main/al_splits.json"
    pairs=pd.read_csv(ROOT/pair_path)
    split=json.loads((ROOT/split_path).read_text(encoding="utf-8"))
    location={d["seed"]:d["flagged_id_locations"]["24"] for d in split}
    palette={"initial":"#39779C","pool":"#B77733","test":"#777777"}
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.9))
    fig.subplots_adjust(left=.105,right=.98,bottom=.21,top=.78,wspace=.40)
    for pos,comparison in enumerate(["duw_minus_uncertainty","duw_minus_maximin"]):
        part=pairs[(pairs.target=="__macro__")&(pairs.comparison==comparison)].sort_values("seed")
        for offset,row in zip(np.linspace(-.15,.15,len(part)),part.itertuples()):
            axes[0].scatter(pos+offset,row.difference,s=20,color=palette[location[row.seed]],alpha=.75,edgecolors="white",linewidths=.3)
        axes[0].scatter(pos+.25,part.difference.mean(),marker="D",s=26,color="black")
    axes[0].set(xticks=[0,1],xticklabels=["DUW - Uncertainty","DUW - Maximin"],ylabel="Paired normalized TestAUC difference",title="Direct comparison with active strategies")
    part=pairs[(pairs.target=="__macro__")&(pairs.comparison=="duw_minus_random")].copy()
    part["location"]=part.seed.map(location)
    for pos,loc in enumerate(["initial","pool","test"]):
        d=part[part.location==loc].sort_values("seed")
        axes[1].scatter(pos+np.linspace(-.16,.16,len(d)),d.difference,color=palette[loc],s=23,alpha=.8,edgecolors="white",linewidths=.3)
        axes[1].scatter(pos+.26,d.difference.mean(),marker="D",s=26,color="black")
    axes[1].set(xticks=[0,1,2],xticklabels=["Initial\n(n = 4)","Pool\n(n = 17)","Test\n(n = 9)"],ylabel="Paired TestAUC: DUW - Random",title="Where was record 24 assigned?")
    for ax in axes:ax.axhline(0,color="#777777",linestyle="--",linewidth=.8)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker="o",linestyle="none",color=color,label="24 in "+loc) for loc,color in palette.items()]
    handles.append(Line2D([],[],marker="D",linestyle="none",color="black",label="Mean difference"))
    fig.legend(handles=handles,ncol=4,loc="upper center",bbox_to_anchor=(.53,.99),frameon=False)
    save(fig,"v2_01_direct_and_record24",list(axes),[pair_path,split_path],
         "原协议30个固定数据配对划分，每个圆点是一项差值，菱形为均值；负值有利于DUW。左：与两种主动策略直接比较。右：按24号所处集合做事后描述，三个子组不是独立物理实验或预设因果比较。两面板纵轴范围不同。")


def model_comparison():
    path="outputs/revision/models/all_metrics.csv";d=pd.read_csv(ROOT/path)
    models=["Mean","OLS","Ridge","ARD_Matern52","Isotropic_Matern52","UniversalKriging_iso_REML"]
    all_values=d[(d.scenario.isin(["real_all24","real_without24"]))&(d.target.isin(["mean_CD","mean_AR"]))&(d.model.isin(models))].r2
    limits=(min(-.48,float(all_values.min())-.05),max(.25,float(all_values.max())+.05))
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.78,wspace=.35)
    for ax,target,title in zip(axes,["mean_CD","mean_AR"],["Mean CD loss","Mean nominal AR"]):
        for offset,scenario,label,color,marker in [(-.10,"real_all24","All 24 records","#39779C","o"),(.10,"real_without24","Without record 24","#B77733","s")]:
            part=d[(d.scenario==scenario)&(d.target==target)].set_index("model").loc[models]
            ax.scatter(np.arange(len(models))+offset,part.r2,color=color,marker=marker,s=30,label=label)
        ax.axhline(0,color="#777777",linestyle="--",linewidth=.8)
        ax.set(xticks=range(6),xticklabels=["Mean","OLS","Ridge","ARD\nGP","Isotropic\nGP","Trend\nUK"],ylabel="Pooled held-out R²",title=title,ylim=limits)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="upper center",bbox_to_anchor=(.53,.99),frameon=False)
    save(fig,"v2_02_models",list(axes),[path],
         "两个固定等权结构平均端点的来源组留出R²。每点来自一套完整折外预测；不是训练R²，也不是六结构R²的均值。端点仅作探索性汇总，不能替代逐结构验收。UK为线性趋势加Matérn残差，未在所有情景稳定胜过Ridge。")


def early_budgets():
    path="outputs/revision/early_al/auc.csv";d=pd.read_csv(ROOT/path)
    d=d[d.target=="normalized_mean"]
    cases=[(s,k,b) for s in ["all24","without24"] for k in ["ARD","isotropic"] for b in [2,4]]
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.7))
    fig.subplots_adjust(left=.23,right=.98,bottom=.16,top=.82,wspace=.30)
    for ax,other in zip(axes,["uncertainty","maximin"]):
        for pos,(scenario,kernel,budget) in enumerate(cases):
            part=d[(d.scenario==scenario)&(d.kernel==kernel)&(d.budget==budget)].pivot(index="seed",columns="strategy",values="test_auc")
            delta=part.duw-part[other];color="#39779C" if kernel=="ARD" else "#B77733"
            ax.scatter(delta,pos+np.linspace(-.14,.14,len(delta)),s=11,color=color,alpha=.5,linewidths=0)
            q=np.quantile(delta,[.25,.5,.75]);ax.plot([q[0],q[2]],[pos,pos],color=color,linewidth=3)
            ax.scatter([delta.mean()],[pos],marker="D",s=22,color="black",zorder=4)
        ax.axvline(0,color="#777777",linestyle="--",linewidth=.8)
        ax.set(yticks=range(8),yticklabels=[f"{'24 records' if s=='all24' else 'Without 24'} | {'ARD' if k=='ARD' else 'ISO'} | B{b}" for s,k,b in cases] if ax is axes[0] else [],
               ylim=(7.6,-.6),xlabel="Paired normalized TestAUC difference",title="DUW - "+other.capitalize())
    fig.text(.54,.95,"Each row: 30 paired splits | diamonds: mean | thick lines: IQR",ha="center",fontsize=8)
    save(fig,"v2_03_early_budgets",list(axes),[path],
         "预算2和4分别重跑，未截断预算8路径。每行30个配对差值，黑菱形为均值、粗线为四分位距，均不是物理总体置信区间。两端点先平均六结构响应，再训练两个GP；因此与旧12响应宏误差不是同一量。负差有利于DUW，各设置之间未形成稳健一致优势。")


def main():
    comparison_and_strata();model_comparison();early_budgets()
    (FIG/"manifest.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding="utf-8")
    insertions=[]
    insertions.append({"after_heading":"4 第24组与响应相关性","blocks":[{"kind":"figure","value":{"path":"outputs/figures/01_ar_data_matrix.png","caption":"原始24条记录的六结构名义AR。第24组的共同高值是来源核查问题；六条AR相关变化不能被计作六份独立证据。"}}]})
    for name,head in [("v2_01_direct_and_record24","6 融合采样的额外收益检验"),("v2_02_models","5 预测基线与模型复杂度"),("v2_03_early_budgets","7 较小预算的探索性回放")]:
        row=next(x for x in MANIFEST if x["name"]==name)
        insertions.append({"after_heading":head,"blocks":[{"kind":"figure","value":{"path":row["png"],"caption":row["caption"]}}]})
    (ROOT/"outputs/revision/document_insertions.json").write_text(json.dumps(insertions,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"figures":len(MANIFEST),"document_figures":len(insertions)}))


if __name__=="__main__":main()
