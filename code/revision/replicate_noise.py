"""重复测量噪声的定位：噪声集中在哪个工艺条件

动机
----
插值误差地图显示一个反直觉现象：按格留出时，**内部格（Platen=40 行）的 CD 误差
最差（27.6%），比边（6.9%）和角（15.8%）都差**。而 Platen=40 恰好是重复测量最多
的一行。

进一步看原始数据：5 个 Coil 水平中有 4 个，P=40 的 CD 值**不落在 P=35 与 P=45 之间**
（例如 Coil=880：0.8735 / 0.8345 / 1.2110），且同格重复测量的差异大得离谱
（Coil=960：1.1435 与 0.8835，差 0.26）。

本脚本把这件事量化：噪声到底集中在哪个条件，量级多少，与信号相比多大。
这决定了后续该"补重复测量"还是"改模型"。

输出：
    outputs/reports/replicate_noise.json
Run:  .venv/bin/python -B code/revision/replicate_noise.py
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
from research.protocol import cell_labels

OUT = ROOT / "outputs/reports"


def main():
    data = load_dataset(ROOT, "real", exclude_ids=(24,))
    X, Y, names = data.X, data.Y, data.target_names
    labels = cell_labels(X[:, :2])
    cd_idx = [j for j, n in enumerate(names) if "CD" in n]
    ar_idx = [j for j, n in enumerate(names) if "AR" in n]

    # 每个格的重复标准差（仅对 n>=2 的格）
    rows = []
    for cell in np.unique(labels):
        members = np.flatnonzero(labels == cell)
        if len(members) < 2:
            continue
        sd_cd = np.array([Y[members, j].std(ddof=0) for j in cd_idx])
        sd_ar = np.array([Y[members, j].std(ddof=0) for j in ar_idx])
        rows.append({"coil": float(X[members[0], 0]),
                     "platen": float(X[members[0], 1]),
                     "n": int(len(members)),
                     "cd_sd_mean": float(sd_cd.mean()),
                     "cd_sd_max": float(sd_cd.max()),
                     "ar_sd_mean": float(sd_ar.mean()),
                     "ar_sd_max": float(sd_ar.max())})

    print("有重复的格及其重复测量标准差：")
    print(f"{'Coil':>6s} {'Platen':>7s} {'n':>2s} {'CD sd(均)':>10s} "
          f"{'CD sd(最大)':>11s} {'AR sd(均)':>10s}")
    for r in sorted(rows, key=lambda r: -r["cd_sd_mean"]):
        print(f"{r['coil']:6.0f} {r['platen']:7.0f} {r['n']:2d} "
              f"{r['cd_sd_mean']:10.4f} {r['cd_sd_max']:11.4f} "
              f"{r['ar_sd_mean']:10.4f}")

    # 按 Platen 水平汇总噪声，并与该水平的信号幅度比较
    print("\n按 Platen 水平汇总（重复标准差 vs 该水平的信号跨度）：")
    summary = {}
    for platen in sorted({r["platen"] for r in rows}):
        sub = [r for r in rows if r["platen"] == platen]
        noise_cd = float(np.mean([r["cd_sd_mean"] for r in sub]))
        # 该 Platen 水平上不同 Coil 之间的 CD 跨度（格均值为基础）
        members = np.flatnonzero(X[:, 1] == platen)
        cell_means = [Y[np.ix_(np.flatnonzero(labels == c), cd_idx)].mean()
                      for c in np.unique(labels[members])]
        signal_cd = float(np.std(cell_means, ddof=0))
        ratio = noise_cd / signal_cd if signal_cd else None
        summary[str(platen)] = {"replicated_cells": len(sub),
                                "cd_within_sd_mean": noise_cd,
                                "cd_between_coil_sd": signal_cd,
                                "noise_to_signal": ratio}
        print(f"  Platen={platen:4.0f}  {len(sub)} 格   "
              f"格内 sd {noise_cd:.4f}   Coil 间 sd {signal_cd:.4f}   "
              f"噪声/信号 = {ratio:.2f}")

    # 噪声占该目标总方差的比例（与噪声上限口径一致）
    total_var = np.array([Y[:, j].var(ddof=0) for j in range(Y.shape[1])])
    within = np.zeros(Y.shape[1])
    for cell in np.unique(labels):
        m = np.flatnonzero(labels == cell)
        if len(m) > 1:
            within += ((Y[m] - Y[m].mean(axis=0)) ** 2).sum(axis=0)
    within_var_share = within / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    print("\n格内噪声占总平方和的比例（= 1 - 噪声上限）：")
    for j, name in enumerate(names):
        print(f"  {name:12s} {within_var_share[j]:6.1%}  → 上限 "
              f"{1 - within_var_share[j]:.3f}")

    # P=40 是否落在 P=35/45 之间（单调性检查）
    print("\n沿 Platen 的单调性（CD loss，3-3 结构）：")
    non_monotone = []
    for coil in sorted({float(v) for v in X[:, 0]}):
        values = {}
        for platen in (35.0, 40.0, 45.0):
            m = np.flatnonzero((X[:, 0] == coil) & (X[:, 1] == platen))
            if len(m):
                values[platen] = float(Y[m, 0].mean())
        if len(values) < 3:
            continue
        lo, hi = min(values[35.0], values[45.0]), max(values[35.0], values[45.0])
        inside = lo <= values[40.0] <= hi
        if not inside:
            non_monotone.append(coil)
        print(f"  Coil={coil:5.0f}  P35 {values[35.0]:.4f}  "
              f"P40 {values[40.0]:.4f}  P45 {values[45.0]:.4f}   "
              f"{'符合单调' if inside else '违反单调'}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "replicate_noise.json"
    path.write_text(json.dumps({
        "replicated_cells": rows,
        "by_platen": summary,
        "within_variance_share": dict(zip(names, within_var_share.tolist())),
        "noise_ceiling_per_target": dict(zip(names, (1 - within_var_share).tolist())),
        "non_monotone_coil_levels": non_monotone,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {path}")
    return rows


if __name__ == "__main__":
    main()
