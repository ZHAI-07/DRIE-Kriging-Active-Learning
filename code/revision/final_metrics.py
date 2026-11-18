"""最终指标：把方法折算成 DRIE 的工程数字

项目原本的优化目标是"用尽量少的工艺实验，把 6 个结构的 CD loss 与 AR 预测准"。
此前所有比较都用 R² / nRMSE 这类无量纲量，工程上不直接可读。本脚本输出：

1. 按格留出（15 折，真正没见过的 Coil×Platen 格）下的逐目标误差，同时给出
   **表单位 RMSE** 与**相对误差百分比**（RMSE / 目标均值）；
2. 噪声上限 `1 - SS_within/SS_total` 与"占上限比例"，说明还剩多少可争取空间；
3. 对设计格做簇自助（cluster bootstrap）得到的置信区间——15 个格是小样本，
   点估计必须配区间；
4. 样本效率曲线：从 6 点开始每多做一个实验，留出误差如何下降，用于回答
   "要达到某精度需要多少实验"；
5. 90% 全共形覆盖率（95% 在 n_train<=14 时不可达，见
   `formal/Formal/Conformal.lean` 的 conformal_95_rank_exceeds_n）。

CD 的单位在数据声明里是 null（`units.note` 明确"不自动沿用历史文件单位"），
因此不写 nm，只写"表单位"与相对百分比；AR 是无量纲，直接给数值误差。

Run:  .venv/bin/python -B code/revision/final_metrics.py
Writes: outputs/reports/final_metrics.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))

from research.data import load_dataset
from research.kernel_gp import FixedKernelGP, FrozenScaler, conformal_feasible
from research.protocol import (cell_labels, leave_one_cell_out, noise_ceiling,
                               pooled_metrics)

OUT = ROOT / "outputs/reports"
TEST_IDS = (1, 5, 12, 14, 16, 17)
N_BOOT = 1000
COLS = [0, 1]


# ------------------------------------------------------------------ 模型
def fit_predict_kernel(X_train, Y_train, X_test):
    """本项目选定方案：2D 固定核 GP，逐目标独立拟合"""
    out = np.empty((len(X_test), Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        sc = FrozenScaler(X_train, Y_train[:, j])
        gp = FixedKernelGP(sc).fit(X_train, Y_train[:, j])
        out[:, j] = gp.predict(X_test)
    return out


def fit_predict_ridge(X_train, Y_train, X_test):
    """项目原有线性基线（全 4 维输入，标准化后 Ridge alpha=1）"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(X_train)
    ym, ys = Y_train.mean(axis=0), Y_train.std(axis=0, ddof=0)
    ys = np.where(ys > 1e-12, ys, 1.0)
    model = Ridge(alpha=1.0).fit(xs.transform(X_train), (Y_train - ym) / ys)
    return model.predict(xs.transform(X_test)) * ys + ym


def fit_predict_mean(X_train, Y_train, X_test):
    return np.tile(Y_train.mean(axis=0), (len(X_test), 1))


def fit_predict_ard(X_train, Y_train, X_test):
    """项目原有 GP 基线：ARD Matern52 + 白噪声，逐目标优化超参"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, Matern, WhiteKernel)
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(X_train)
    Z, Zt = xs.transform(X_train), xs.transform(X_test)
    out = np.empty((len(X_test), Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        ym, ys = float(Y_train[:, j].mean()), float(Y_train[:, j].std(ddof=0)) or 1.0
        k = (ConstantKernel(1.0, (1e-3, 1e3))
             * Matern(np.ones(X_train.shape[1]), (1e-2, 10.0), nu=2.5)
             + WhiteKernel(1e-3, (1e-6, 1e-1)))
        gp = GaussianProcessRegressor(kernel=k, alpha=1e-3, normalize_y=False,
                                      random_state=j).fit(Z, (Y_train[:, j] - ym) / ys)
        out[:, j] = gp.predict(Zt) * ys + ym
    return out


MODELS = {"FixedKernelGP 2D（本方案）": fit_predict_kernel,
          "Ridge(alpha=1) 4D（原基线）": fit_predict_ridge,
          "ARD GP 4D（原基线）": fit_predict_ard,
          "Mean（平凡基线）": fit_predict_mean}


# --------------------------------------------------------------- 工程指标
def engineering_rows(Y_true, Y_hat, names):
    """逐目标给出 R²、nRMSE、表单位 RMSE 与相对误差百分比"""
    rows = []
    for j, name in enumerate(names):
        truth, pred = Y_true[:, j], Y_hat[:, j]
        err = pred - truth
        rmse = float(np.sqrt((err ** 2).mean()))
        ss_res = float((err ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        sd = float(truth.std(ddof=0)) or 1.0
        mean_abs = float(np.abs(truth).mean()) or 1.0
        rows.append({"target": name, "r2": 1.0 - ss_res / ss_tot,
                     "nrmse": rmse / sd, "rmse_units": rmse,
                     "rel_error_pct": 100.0 * rmse / mean_abs,
                     "mean_units": float(truth.mean())})
    return rows


def macro(rows, names):
    """宏平均，并按 CD / AR 分组"""
    cd = [r for r, n in zip(rows, names) if "CD" in n]
    ar = [r for r, n in zip(rows, names) if "AR" in n]
    def agg(sub):
        return {"r2": float(np.mean([r["r2"] for r in sub])),
                "nrmse": float(np.mean([r["nrmse"] for r in sub])),
                "rmse_units": float(np.mean([r["rmse_units"] for r in sub])),
                "rel_error_pct": float(np.mean([r["rel_error_pct"] for r in sub]))}
    return {"all": agg(rows), "cd": agg(cd), "ar": agg(ar)}


def predict_all(fn, X, Y, splits):
    Y_hat = np.full_like(Y, np.nan, dtype=float)
    for train, test in splits:
        Y_hat[test] = np.asarray(fn(X[train], Y[train], X[test]), float)
    return Y_hat


def cluster_bootstrap(Y, Y_hat, labels, names, n_boot=N_BOOT, seed=0):
    """对设计格做簇自助，给宏平均 R² / 相对误差的置信区间

    单元是"格"而不是记录：同格记录高度相关，按记录抽样会低估方差。
    """
    rng = np.random.default_rng(seed)
    cells = np.unique(labels)
    boot = {"r2": [], "rel_cd": [], "rel_ar": [], "nrmse": []}
    for _ in range(n_boot):
        pick = rng.choice(cells, size=len(cells), replace=True)
        idx = np.concatenate([np.flatnonzero(labels == c) for c in pick])
        rows = engineering_rows(Y[idx], Y_hat[idx], names)
        agg = macro(rows, names)
        boot["r2"].append(agg["all"]["r2"])
        boot["nrmse"].append(agg["all"]["nrmse"])
        boot["rel_cd"].append(agg["cd"]["rel_error_pct"])
        boot["rel_ar"].append(agg["ar"]["rel_error_pct"])
    return {k: {"bootstrap_mean": float(np.mean(v)),
                "ci95": [float(np.percentile(v, 2.5)),
                         float(np.percentile(v, 97.5))],
                "p_above_zero": float(np.mean(np.array(v) > 0))}
            for k, v in boot.items()}


def coverage(Y, X2, labels, alpha, names, use_cpp=True):
    """按格留出的全共形覆盖率与区间宽度（表单位除以目标 sd）"""
    hits, widths = [], {n: [] for n in names}
    for train, test in leave_one_cell_out(labels):
        for j, name in enumerate(names):
            sc = FrozenScaler(X2[train], Y[train, j])
            gp = FixedKernelGP(sc).fit(X2[train], Y[train, j])
            for row in test:
                lo, hi = gp.conformal_interval(Y[train, j], X2[row],
                                               alpha=alpha, use_cpp=use_cpp)
                hit = bool(lo <= Y[row, j] <= hi)
                hits.append(hit)
                sd = float(Y[train, j].std(ddof=0)) or 1.0
                widths[name].append((hi - lo) / sd if np.isfinite(hi - lo) else np.inf)
    return {"alpha": alpha, "coverage": float(np.mean(hits)),
            "n_cells_checked": len(hits),
            "mean_width_sd": float(np.mean([w for w in widths.values() for w in w]))}


def learning_curve(X2, Y, ids, names, max_steps=8):
    """从 6 点 D-optimal 起点，每多做一个实验的留出误差"""
    from scipy.spatial.distance import cdist
    test_idx = np.array([int(np.flatnonzero(ids == t)[0]) for t in TEST_IDS])
    pool = [i for i in range(len(X2)) if i not in set(test_idx)]
    labels_pool = cell_labels(X2[pool])
    # 每个格取首条记录作为候选，构成 D-optimal 选点基础
    cells = {}
    for i, lab in zip(pool, labels_pool):
        cells.setdefault(int(lab), []).append(i)
    reps = [v[0] for v in cells.values()]
    # 枚举六点设计，最大化 det(K)
    import itertools
    best, best_det = None, -np.inf
    for combo in itertools.combinations(reps, 6):
        sc = FrozenScaler(X2[list(combo)], None)
        Z = sc.x(X2[list(combo)])
        K = FixedKernelGP(sc).gram_train(Z)
        det = float(np.linalg.det(K))
        if det > best_det:
            best, best_det = combo, det
    train = list(best)
    curve = []
    for step in range(max_steps + 1):
        Y_hat = np.empty((len(test_idx), Y.shape[1]))
        for j in range(Y.shape[1]):
            sc = FrozenScaler(X2[train], Y[train, j])
            gp = FixedKernelGP(sc).fit(X2[train], Y[train, j])
            Y_hat[:, j] = gp.predict(X2[test_idx])
        rows = engineering_rows(Y[test_idx], Y_hat, names)
        agg = macro(rows, names)
        curve.append({"n_train": len(train),
                      "nrmse": agg["all"]["nrmse"],
                      "rel_cd_pct": agg["cd"]["rel_error_pct"],
                      "rel_ar_pct": agg["ar"]["rel_error_pct"]})
        if step == max_steps:
            break
        sc = FrozenScaler(X2[train], None)
        Z = sc.x(X2[train])
        gp = FixedKernelGP(sc).fit(X2[train], Y[list(train), 0])
        Zc = sc.x(X2[np.array([i for i in pool if i not in set(train)])])
        ks = gp.gram(Zc, Z)
        v = cho_solve(gp._chol, ks.T)
        var = (gp.signal + gp.noise) - np.einsum("ij,ji->i", ks, v)
        sd = np.sqrt(np.maximum(var, 1e-12))
        remaining = [i for i in pool if i not in set(train)]
        train.append(int(remaining[int(np.argmax(sd))]))
    return curve


def main():
    data = load_dataset(ROOT, "real", exclude_ids=(24,))
    X, Y, names = data.X, data.Y, data.target_names
    X2 = X[:, COLS]
    labels = cell_labels(X2)
    splits = leave_one_cell_out(labels)
    ceiling = noise_ceiling(Y, labels)

    # 每个模型用各自声明的输入集：本方案只用 Coil+Platen，原基线沿用全 4 维，
    # 否则"2D 固定核"会被悄悄跑成 4D isotropic（实测会从 +0.209 掉到 +0.116）。
    inputs = {"FixedKernelGP 2D（本方案）": X2,
              "Ridge(alpha=1) 4D（原基线）": X,
              "ARD GP 4D（原基线）": X,
              "Mean（平凡基线）": X}

    report = {"data": {"records": int(len(X)), "targets": len(names),
                       "cells": int(len(np.unique(labels))),
                       "cd_units": "um (user-confirmed)",
                       "ar_units": "dimensionless"},
              "protocol": "leave-one-cell-out on (Coil, Platen), 15 folds",
              "noise_ceiling": {"macro": ceiling["macro"],
                                "per_target": dict(zip(names, ceiling["per_target"]))},
              "models": {}}

    print(f"{'模型':30s} {'宏R²':>7s} {'宏nRMSE':>8s} {'CD相对误差':>10s} {'AR相对误差':>10s}")
    preds = {}
    for name, fn in MODELS.items():
        t0 = time.perf_counter()
        Y_hat = predict_all(fn, inputs[name], Y, splits)
        preds[name] = Y_hat
        rows = engineering_rows(Y, Y_hat, names)
        agg = macro(rows, names)
        cei = float(np.mean([ceiling["per_target"][i] for i in range(len(names))]))
        entry = {"per_target": rows, "macro": agg,
                 "fraction_of_ceiling": agg["all"]["r2"] / cei if cei else None,
                 "headroom_used_pct": (100.0 * agg["all"]["r2"] / cei) if cei else None,
                 "seconds": time.perf_counter() - t0}
        report["models"][name] = entry
        print(f"{name:30s} {agg['all']['r2']:+7.3f} {agg['all']['nrmse']:8.3f} "
              f"{agg['cd']['rel_error_pct']:9.1f}% {agg['ar']['rel_error_pct']:9.1f}%")

    # 自助区间只给两个主模型，控制运行时间
    print("\n簇自助置信区间（按设计格重采样，B=1000）：")
    for name in ("FixedKernelGP 2D（本方案）", "Ridge(alpha=1) 4D（原基线）"):
        ci = cluster_bootstrap(Y, preds[name], labels, names)
        report["models"][name]["bootstrap"] = ci
        point = report["models"][name]["macro"]["all"]["r2"]
        print(f"  {name:30s} 宏R² {point:+.3f} "
              f"[{ci['r2']['ci95'][0]:+.3f}, {ci['r2']['ci95'][1]:+.3f}]  "
              f"P(R²>0)={ci['r2']['p_above_zero']:.2f}  "
              f"CD {ci['rel_cd']['bootstrap_mean']:.1f}%  "
              f"AR {ci['rel_ar']['bootstrap_mean']:.1f}%")

    # 样本效率
    curve = learning_curve(X2, Y, data.ids, names)
    report["learning_curve"] = curve
    print("\n样本效率（D-optimal 起点 + 最大方差采集，6 条固定测试记录）：")
    print(f"{'n_train':>8s} {'宏nRMSE':>8s} {'CD相对误差':>10s} {'AR相对误差':>10s}")
    for point in curve:
        print(f"{point['n_train']:8d} {point['nrmse']:8.3f} "
              f"{point['rel_cd_pct']:9.1f}% {point['rel_ar_pct']:9.1f}%")

    # 覆盖率：90% 可用，95% 在 n_train<=14 时不可达
    print("\n全共形覆盖率：")
    for alpha in (0.10, 0.05):
        n_cal = len(X) - 1 if alpha == 0.10 else len(X) - 1
        feasible = conformal_feasible(n_cal, alpha)
        if not feasible:
            report.setdefault("conformal", {})[f"alpha_{alpha}"] = {
                "feasible": False,
                "reason": f"rank {math.ceil((n_cal+1)*(1-alpha))} > n={n_cal}"}
            print(f"  alpha={alpha}: 不可达（rank > n），按定理跳过")
            continue
        cov = coverage(Y, X2, labels, alpha, names)
        report.setdefault("conformal", {})[f"alpha_{alpha}"] = cov
        print(f"  alpha={alpha}: 覆盖 {cov['coverage']:.3f}  "
              f"平均宽度 {cov['mean_width_sd']:.2f} 目标sd  "
              f"（{cov['n_cells_checked']} 个单元格）")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "final_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {path}")
    return report


if __name__ == "__main__":
    main()
