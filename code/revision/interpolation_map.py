"""插值误差地图：按格位置分解误差，并量化外推可信度

背景
----
这份数据的 4 个工艺输入里，只有 Coil（5 个水平：840/880/920/960/1000）与
Platen（3 个水平：35/40/45）携带信息，且 15 个组合**全部测过**——是完整的
5x3 全因子，一格不缺。因此按格留出得到的误差不是"缺数据"，而是纯粹的
格内插值误差。

既然如此，工程上真正有用的问题不是"再测哪个格"，而是：

1. 误差在网格不同位置是否不同（角、边、内部）？
2. 把条件推出当前测过的范围时，预测会多快退化回均值？

第 2 点无法用新数据验证，但可以用核本身量化：Matern 5/2 的相关在一步网格
之外剩多少，直接决定外推值还有多少信息含量。这属于模型内在性质，不是对
外推精度的实测声称。

输出：
    outputs/reports/interpolation_map.json
Run:  .venv/bin/python -B code/revision/interpolation_map.py
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
from research.kernel_gp import FixedKernelGP, FrozenScaler, matern52
from research.protocol import cell_labels

OUT = ROOT / "outputs/reports"
COLS = [0, 1]


def position_class(coil, platen, coil_levels, platen_levels):
    """把格分成角、边、内部三类"""
    ci = coil_levels.index(coil)
    pi = platen_levels.index(platen)
    on_coil_edge = ci in (0, len(coil_levels) - 1)
    on_plat_edge = pi in (0, len(platen_levels) - 1)
    if on_coil_edge and on_plat_edge:
        return "corner"
    if on_coil_edge or on_plat_edge:
        return "edge"
    return "interior"


def neighbor_count(coil, platen, coil_levels, platen_levels):
    """该格在 5x3 网格内的正交邻居数（2..4）"""
    ci = coil_levels.index(coil)
    pi = platen_levels.index(platen)
    return (int(ci > 0) + int(ci < len(coil_levels) - 1)
            + int(pi > 0) + int(pi < len(platen_levels) - 1))


def main():
    data = load_dataset(ROOT, "real", exclude_ids=(24,))
    X, Y, names = data.X, data.Y, data.target_names
    X2 = X[:, COLS]
    labels = cell_labels(X2)
    coil_levels = sorted({float(v) for v in X[:, 0]})
    platen_levels = sorted({float(v) for v in X[:, 1]})

    rows = []
    for cell in np.unique(labels):
        test = np.flatnonzero(labels == cell)
        train = np.flatnonzero(labels != cell)
        Y_hat = np.empty((len(test), Y.shape[1]))
        for j in range(Y.shape[1]):
            sc = FrozenScaler(X2[train], Y[train, j])
            gp = FixedKernelGP(sc).fit(X2[train], Y[train, j])
            Y_hat[:, j] = gp.predict(X2[test])
        coil, platen = float(X2[test[0], 0]), float(X2[test[0], 1])
        cd_rel, ar_rel = [], []
        for j, name in enumerate(names):
            err = Y_hat[:, j] - Y[test, j]
            rmse = float(np.sqrt((err ** 2).mean()))
            scale = float(np.abs(Y[test, j]).mean()) or 1.0
            (cd_rel if "CD" in name else ar_rel).append(100.0 * rmse / scale)
        rows.append({
            "coil": coil, "platen": platen, "n_replicates": int(len(test)),
            "position": position_class(coil, platen, coil_levels, platen_levels),
            "neighbors": neighbor_count(coil, platen, coil_levels, platen_levels),
            "cd_rel_pct": float(np.mean(cd_rel)),
            "ar_rel_pct": float(np.mean(ar_rel)),
        })

    print(f"{'Coil':>6s} {'Platen':>7s} {'n':>2s} {'位置':>9s} {'邻居':>5s} "
          f"{'CD相对误差':>10s} {'AR相对误差':>10s}")
    for r in sorted(rows, key=lambda r: (r["position"], -r["cd_rel_pct"])):
        print(f"{r['coil']:6.0f} {r['platen']:7.0f} {r['n_replicates']:2d} "
              f"{r['position']:>9s} {r['neighbors']:5d} "
              f"{r['cd_rel_pct']:9.1f}% {r['ar_rel_pct']:9.2f}%")

    # 按位置分类汇总
    groups = {}
    for r in rows:
        groups.setdefault(r["position"], []).append(r)
    print("\n按位置分类：")
    summary = {}
    for pos, sub in sorted(groups.items()):
        cd = float(np.mean([r["cd_rel_pct"] for r in sub]))
        ar = float(np.mean([r["ar_rel_pct"] for r in sub]))
        summary[pos] = {"cells": len(sub), "cd_rel_pct": cd, "ar_rel_pct": ar}
        print(f"  {pos:9s} {len(sub):2d} 格   CD {cd:5.1f}%   AR {ar:5.2f}%")

    # 误差与邻居数的关系（邻居越少越接近边界）
    print("\n按正交邻居数：")
    by_nb = {}
    for r in rows:
        by_nb.setdefault(r["neighbors"], []).append(r)
    nb_summary = {}
    for nb, sub in sorted(by_nb.items()):
        cd = float(np.mean([r["cd_rel_pct"] for r in sub]))
        nb_summary[str(nb)] = {"cells": len(sub), "cd_rel_pct": cd}
        print(f"  {nb} 邻居  {len(sub):2d} 格   CD {cd:5.1f}%")

    # 外推可信度：一步/两步网格之外，核相关还剩多少
    sc_x = FrozenScaler(X2, None)
    sd_coil = float(sc_x.x(X2)[:, 0].std(ddof=0))
    step_coil = (coil_levels[1] - coil_levels[0]) / float(sc_x.x_sd[0])
    step_plat = (platen_levels[1] - platen_levels[0]) / float(sc_x.x_sd[1])
    print("\n外推可信度（核相关，一步网格之外）：")
    extrap = {}
    for label, dist in (("沿 Coil 一步 (Δ40)", step_coil),
                        ("沿 Platen 一步 (Δ5)", step_plat),
                        ("沿 Coil 两步 (Δ80)", 2 * step_coil),
                        ("沿对角一步", float(np.hypot(step_coil, step_plat)))):
        s = np.sqrt(5.0) * dist / 1.2
        k = float(matern52(s))
        extrap[label] = {"distance_std": float(dist), "correlation": k}
        print(f"  {label:20s} 距离 {dist:5.3f} 标准化单位   相关 {k:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "interpolation_map.json"
    path.write_text(json.dumps({
        "design": {"coil_levels": coil_levels, "platen_levels": platen_levels,
                   "complete_factorial": len(rows) == len(coil_levels) * len(platen_levels),
                   "replicated_cells": sum(1 for r in rows if r["n_replicates"] > 1)},
        "per_cell": rows, "by_position": summary, "by_neighbors": nb_summary,
        "extrapolation_correlation": extrap,
        "protocol": "leave-one-cell-out on the complete 5x3 factorial",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {path}")
    return rows


if __name__ == "__main__":
    main()
