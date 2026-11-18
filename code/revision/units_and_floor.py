"""以 μm 为单位给出绝对误差，并计算距"噪声下限"还剩多少

两条外部信息改变了结论的落点：

1. CD loss 的单位是 **μm**（用户确认）。此前源表单位未声明，报告只能用"表单位"
   与相对百分比；现在可以给出绝对误差，与工艺容差直接比较。
2. P=40 那几次重复实验之间**没有**未记录的变量（腔体状态、时序、片间漂移均无）。
   因此该行 9%–16% 的跑间离散不是"漏了一个解释变量"，而是这套输入下的不可约噪声
   （工艺随机性 + 计量重复性）。

于是可以回答工程上最关心的问题：**在给定重复性的前提下，任何模型最多能做到多好，
我们现在离它有多远？**

噪声下限的定义：若某预测器完全知道每个设计格的真值（格均值），它的留出 RMSE 就是
`sqrt(SS_within / N)`。这是可达精度的下界，不含任何模型假设——格内差异既然是噪声，
就没有任何函数能在格内分辨它。

输出：
    outputs/reports/units_and_floor.json
Run:  .venv/bin/python -B code/revision/units_and_floor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))

from research.data import load_dataset
from research.protocol import cell_labels, leave_one_cell_out
from research.kernel_gp import FixedKernelGP, FrozenScaler

OUT = ROOT / "outputs/reports"
COLS = [0, 1]


def model_predictions(X2, Y, splits):
    Y_hat = np.full_like(Y, np.nan, dtype=float)
    for train, test in splits:
        for j in range(Y.shape[1]):
            sc = FrozenScaler(X2[train], Y[train, j])
            gp = FixedKernelGP(sc).fit(X2[train], Y[train, j])
            Y_hat[test, j] = gp.predict(X2[test])
    return Y_hat


def within_cell_scatter(Y, labels):
    """格内平方和、格内自由度，用于噪声下限与方差齐性检验"""
    ss, df, groups = 0.0, 0, []
    for cell in np.unique(labels):
        m = np.flatnonzero(labels == cell)
        if len(m) > 1:
            dev = Y[m] - Y[m].mean(axis=0)
            ss += float((dev ** 2).sum())
            df += len(m) - 1
            groups.append(dev)
    return ss, df, groups


def variance_homogeneity(Y, labels, n_boot=5000, seed=0):
    """格内方差的齐性：Levene + 自助（小样本功效极低，必须连同功效一起报）

    注意 Coil=840/Platen=40 两次 CD 读数完全相同（0.8350、0.8350），组内 sd 恰为 0；
    这类退化组会让 sd 极差比发散，因此把 0 单独标出，不计入比值。
    """
    from scipy.stats import levene

    groups = []
    for cell in np.unique(labels):
        m = np.flatnonzero(labels == cell)
        if len(m) > 1:
            groups.append(Y[m, 0])
    stat, p = levene(*groups)
    sds = np.array([g.std(ddof=1) for g in groups])
    degenerate = [float(g[0]) for g in groups if g.std(ddof=1) < 1e-12]
    positive = sds[sds > 1e-12]
    observed_ratio = float(positive.max() / positive.min()) if len(positive) > 1 else float("nan")

    rng = np.random.default_rng(seed)
    pooled_sd = float(np.sqrt(np.mean(positive ** 2)))
    sizes = [len(g) for g in groups]
    count = 0
    for _ in range(n_boot):
        sim = [rng.normal(0.0, pooled_sd, s) for s in sizes]
        sim_sds = np.array([s.std(ddof=1) for s in sim])
        sim_pos = sim_sds[sim_sds > 1e-12]
        if len(sim_pos) > 1 and sim_pos.max() / sim_pos.min() >= observed_ratio:
            count += 1
    return {"levene_stat": float(stat), "levene_p": float(p),
            "sd_max_over_min": observed_ratio,
            "degenerate_groups_value": degenerate,
            "group_sds": sds.tolist(),
            "bootstrap_p_under_homogeneous": count / n_boot,
            "group_sizes": sizes,
            "power_note": "每组仅 2–4 个点，方差齐性检验功效极低；"
                          "结论只能读作『未发现结构』，不能读作『已确认同质』"}


def main():
    data = load_dataset(ROOT, "real", exclude_ids=(24,))
    X, Y, names = data.X, data.Y, data.target_names
    X2 = X[:, COLS]
    labels = cell_labels(X2)
    splits = leave_one_cell_out(labels)
    Y_hat = model_predictions(X2, Y, splits)

    report = {"units": {"cd": "um (user-confirmed)", "ar": "dimensionless"},
              "rationale": "P=40 重复实验之间无未记录变量（用户确认），故该行离散为"
                           "不可约噪声；噪声下限 = sqrt(SS_within / N)",
              "per_target": {}}

    print(f"{'目标':12s} {'均值':>8s} {'模型RMSE':>9s} {'噪声下限':>9s} "
          f"{'倍数':>6s} {'可达改进':>8s}")
    for j, name in enumerate(names):
        truth, pred = Y[:, j], Y_hat[:, j]
        err = pred - truth
        rmse = float(np.sqrt((err ** 2).mean()))
        ss_res = float((err ** 2).sum())
        ss_within, _, _ = within_cell_scatter(Y[:, j:j + 1], labels)
        n = len(truth)
        floor = float(np.sqrt(ss_within / n))
        ratio = rmse / floor if floor > 0 else None
        # 若达到下限，R² 会到 1 - SS_within/SS_total
        ss_total = float(((truth - truth.mean()) ** 2).sum())
        ceiling = 1.0 - ss_within / ss_total
        report["per_target"][name] = {
            "unit": "um" if "CD" in name else "dimensionless",
            "mean": float(truth.mean()), "sd": float(truth.std(ddof=0)),
            "model_rmse": rmse, "noise_floor_rmse": floor,
            "model_over_floor": ratio,
            "potential_rmse_reduction_pct": 100.0 * (1.0 - floor / rmse) if rmse else None,
            "r2_ceiling": ceiling}
        print(f"{name:12s} {truth.mean():8.4f} {rmse:9.4f} {floor:9.4f} "
              f"{ratio:6.2f} {100 * (1 - floor / rmse):7.1f}%")

    cd = [v for k, v in report["per_target"].items() if "CD" in k]
    ar = [v for k, v in report["per_target"].items() if "AR" in k]
    report["macro"] = {
        "cd": {"mean_um": float(np.mean([v["mean"] for v in cd])),
               "model_rmse_um": float(np.mean([v["model_rmse"] for v in cd])),
               "noise_floor_um": float(np.mean([v["noise_floor_rmse"] for v in cd])),
               "model_over_floor": float(np.mean([v["model_over_floor"] for v in cd]))},
        "ar": {"mean": float(np.mean([v["mean"] for v in ar])),
               "model_rmse": float(np.mean([v["model_rmse"] for v in ar])),
               "noise_floor": float(np.mean([v["noise_floor_rmse"] for v in ar])),
               "model_over_floor": float(np.mean([v["model_over_floor"] for v in ar]))}}
    print("\n宏平均：")
    print(f"  CD  loss  均值 {report['macro']['cd']['mean_um']:.4f} μm   "
          f"误差 {report['macro']['cd']['model_rmse_um']:.4f} μm "
          f"({report['macro']['cd']['model_rmse_um']*1000:.0f} nm)   "
          f"下限 {report['macro']['cd']['noise_floor_um']*1000:.0f} nm   "
          f"倍数 {report['macro']['cd']['model_over_floor']:.2f}")
    print(f"  AR        均值 {report['macro']['ar']['mean']:.3f}   "
          f"误差 {report['macro']['ar']['model_rmse']:.4f}   "
          f"下限 {report['macro']['ar']['noise_floor']:.4f}   "
          f"倍数 {report['macro']['ar']['model_over_floor']:.2f}")

    # P=40 行的噪声是否同质（回答"是否还有结构"）
    print("\nP=40 行格内方差齐性检验（回答：噪声是否还有结构）：")
    homo = variance_homogeneity(Y, labels)
    report["p40_variance_homogeneity"] = homo
    print(f"  各组 sd {[round(s, 4) for s in homo['group_sds']]}   "
          f"样本量 {homo['group_sizes']}")
    print(f"  Levene p = {homo['levene_p']:.3f}   "
          f"非退化组的 sd 极差比 = {homo['sd_max_over_min']:.1f}")
    print(f"  退化组（两次读数完全相同）的值 {homo['degenerate_groups_value']}"
          f" —— 是数据事实，不计入比值")
    print(f"  单一噪声源自助下，极差比不亚于观测值的频率 = "
          f"{homo['bootstrap_p_under_homogeneous']:.3f}")
    print(f"  {homo['power_note']}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "units_and_floor.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {path}")
    return report


if __name__ == "__main__":
    main()
