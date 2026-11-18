"""小样本评估协议：按格留出、噪声上限、置换对照、伪重复检查

本模块把本项目 2026-11 审查中确立的四条评估纪律固化为可复用部件。每一条都对应
一次实测踩坑，目的不是提供新算法，而是让"比较两个模型/两种采集规则"这类判断
不再重复同类错误。

背景（全部为实测，非常识推定）：

1. 按格留出。23条记录只占15个(Coil, Platen)格（5格有重复：4/3/2/2/2条），
   而4维输入下23条互异。按记录留出时，被留出记录的同格重复样本仍在训练集里，
   模型等于在回答"格均值"。实测同一模型按记录留出 R²=+0.410、按格留出
   R²=+0.209，线性基线更从+0.122掉到+0.012。所以留出单元必须按物理重复格
   定义，且报告时必须写明留出单元是哪一种。

2. 噪声上限。格内差异是测量噪声，而(Coil, Platen)的函数在同一格内必为常数，
   故任何此类模型的最优R²上限为 1 - SS_within/SS_total。实测CD上限0.900、
   AR上限0.868、全体0.884。脱离上限谈"R²=0.21偏弱"没有意义。

3. 置换对照。给噪声维一个极长长度尺度（tau=10）并非"移除"它们，而是让恰好
   共享同一取值的记录拿到更高核权重。实测4D flat模型 R²=+0.242 优于2D的
   +0.209，但把SF6/压力两列按行置换后仍得+0.226：约一半增益来自正则化几何，
   与信息无关。凡"加特征/加维度带来提升"的结论都必须配此对照。

4. 伪重复检查。固定设计+重复记录选择不构成独立重复：某个D-optimal六格设计中
   5格各只有1条记录，30个种子实际只产生2个不同初始集，配对符号检验有效样本
   量是2，曾因此得出假 p<0.001。配对检验前必须先数独立单元。

对外接口均为纯函数，不读文件、不写文件、不依赖具体模型。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# 判断"同一设计点"的默认输入列：Coil 与 Platen
DEFAULT_CELL_COLUMNS = (0, 1)


# ---------------------------------------------------------------- 格结构
def cell_labels(X, columns=DEFAULT_CELL_COLUMNS):
    """按指定输入列把记录归入设计格，返回整数标签数组

    参数：
        X: (n, d) 输入矩阵
        columns: 参与定义"格"的列索引，默认Coil+Platen

    返回：
        (n,) int 数组，同一格取值相同，编号按首次出现顺序
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵")
    columns = tuple(columns)
    if not columns:
        raise ValueError("columns 不能为空")
    index = {}
    labels = np.empty(len(X), dtype=int)
    for row_id, row in enumerate(X):
        key = tuple(float(row[c]) for c in columns)
        labels[row_id] = index.setdefault(key, len(index))
    return labels


def cell_counts(labels):
    """每个格的记录数，按标签顺序"""
    labels = np.asarray(labels, int)
    return np.bincount(labels)


def cell_table(X, labels, ids=None):
    """逐格清点：输入坐标、记录数、成员ID

    用于在报告里写明设计结构（几格、哪些格有重复），这是判断能否按记录留出的
    前提。
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    rows = []
    for cell in range(int(labels.max()) + 1 if len(labels) else 0):
        members = np.flatnonzero(labels == cell)
        if not len(members):
            continue
        rows.append({
            "cell": cell,
            "inputs": [float(v) for v in X[members[0]]],
            "count": int(len(members)),
            "member_ids": None if ids is None else [int(ids[i]) for i in members],
        })
    return rows


def leave_one_cell_out(labels):
    """留一格交叉验证的切分，返回 [(train_idx, test_idx), ...]

    每次留出整格（含该格全部重复记录），训练集不含任何同格记录，因此模型的
    输入在训练集中不出现，评价的是真正的插值/外推误差。
    """
    labels = np.asarray(labels, int)
    if not len(labels):
        return []
    splits = []
    for cell in np.unique(labels):
        test = np.flatnonzero(labels == cell)
        train = np.flatnonzero(labels != cell)
        if len(train) == 0:
            continue
        splits.append((train, test))
    return splits


# ---------------------------------------------------------------- 噪声上限
def noise_ceiling(Y, labels):
    """格内噪声决定的可达R²上限：1 - SS_within / SS_total，逐目标给出

    论证：(Coil, Platen)的函数在同一格内必为常数，故任何此类预测器在同一格内
    只能给一个值，而格内最优常数是格均值（逐格的一维最小二乘），残差即格内
    平方和。加总即得上限。这里不涉及可交换性等假设，是代数恒等式。

    参数：
        Y: (n, T) 响应矩阵
        labels: (n,) 格标签

    返回：
        dict，含 per_target（列表）、macro（均值）、以及 CD/AR 分组均值（若目标名
        未提供则不分组）
    """
    Y = np.asarray(Y, float)
    labels = np.asarray(labels, int)
    if Y.shape[0] != labels.shape[0]:
        raise ValueError("Y 与 labels 行数不一致")
    total = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    within = np.zeros(Y.shape[1])
    for cell in np.unique(labels):
        members = np.flatnonzero(labels == cell)
        if len(members) > 1:
            block = Y[members]
            within += ((block - block.mean(axis=0)) ** 2).sum(axis=0)
    # 逐目标上限；总平方和为零的目标（常数响应）无定义，用 nan 表示且不触发除零告警
    safe_total = np.where(total > 1e-12, total, np.nan)
    ceiling = 1.0 - within / safe_total
    macro = float(np.nanmean(ceiling)) if np.any(np.isfinite(ceiling)) else float("nan")
    return {"per_target": ceiling.tolist(), "macro": macro,
            "within_ss": within.tolist(), "total_ss": total.tolist(),
            "replicated_cells": int((np.bincount(labels) > 1).sum())}


# ---------------------------------------------------------------- 指标
def pooled_metrics(Y_true, Y_hat):
    """合并留出预测的逐目标 R² / RMSE / nRMSE，附宏平均

    nRMSE 以合并真值的总体标准差归一，与 code/research/experiments.py 的
    eval_scale 口径一致。
    """
    Y_true = np.asarray(Y_true, float)
    Y_hat = np.asarray(Y_hat, float)
    if Y_true.shape != Y_hat.shape:
        raise ValueError("形状不一致")
    error = Y_hat - Y_true
    rmse = np.sqrt((error ** 2).mean(axis=0))
    sd = Y_true.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    ss_res = (error ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    return {"r2": r2.tolist(), "rmse": rmse.tolist(),
            "nrmse": (rmse / sd).tolist(),
            "macro": {"r2": float(np.nanmean(r2)),
                      "rmse": float(rmse.mean()),
                      "nrmse": float((rmse / sd).mean())}}


def fold_nrmse(Y_true, Y_hat, scale):
    """单折 nRMSE，用外部给定的全局标准差归一

    单点或单格的 ss_tot 为0，逐折R²无定义；折级排序必须改用外部尺度。
    scale 必须是预先固定、不随折变化的每目标标准差（如全体数据的 sd），
    否则折间不可比。
    """
    Y_true = np.asarray(Y_true, float)
    Y_hat = np.asarray(Y_hat, float)
    scale = np.asarray(scale, float)
    error = Y_hat - Y_true
    if error.ndim == 1:
        error = error.reshape(-1, 1)
    rmse = np.sqrt((error ** 2).mean(axis=0))
    safe = np.where(scale > 1e-12, scale, 1.0)
    return float(np.mean(rmse / safe))


# ---------------------------------------------------------------- 置换对照
def permutation_control(X, columns, rng):
    """按行置换指定列，保留边际分布、摧毁与响应的关联

    用于分离"加入这些列带来的几何/正则化效应"与"这些列携带的信息"。实测中
    约一半的增益在置换后仍然存在，说明那一半与信息无关。
    """
    X = np.array(X, dtype=float, copy=True)
    for column in columns:
        X[:, column] = X[rng.permutation(len(X)), column]
    return X


# ---------------------------------------------------------------- 伪重复
@dataclass
class Replication:
    """独立单元计数结果

    distinct 远小于 draws 时，配对检验的有效样本量是 distinct 而非 draws。
    """
    draws: int
    distinct: int
    duplicated: bool
    message: str


def count_distinct_initial_sets(initial_sets):
    """数独立初始设计单元，符合独立重复的前提才允许做配对检验

    参数：
        initial_sets: 可迭代，每项是可哈希的初始设计标识（如记录索引的元组）

    返回：
        Replication 报告；duplicated=True 表示 draws 中存在重复单元
    """
    seen = {tuple(sorted(int(i) for i in s)) for s in initial_sets}
    draws = len(list(initial_sets)) if not isinstance(initial_sets, list) else len(initial_sets)
    distinct = len(seen)
    message = (f"{draws} 次抽样中只有 {distinct} 个不同初始集"
               if distinct < draws else f"{draws} 次抽样全部互异")
    return Replication(draws=draws, distinct=distinct,
                       duplicated=distinct < draws, message=message)


def paired_summary(scores_a, scores_b, higher_is_better=False):
    """配对比较：符号检验 + Wilcoxon + 独立单元提示

    同时给出两种检验，因为二者对平局处理与分布形状的敏感度不同：实测同一效应量
    下符号检验从 p=0.005 变到 p=0.200，而 Wilcoxon 稳定在 0.006。只报一个会
    系统性高估或低估稳健性。

    参数：
        scores_a, scores_b: 配对分数；"a 更好"对应更小值（higher_is_better=False）
        higher_is_better: 置 True 表示数值越大越好

    返回：
        dict，含 mean_a/mean_b、delta（a-b）、wins/losses、符号检验与 Wilcoxon 的 p
    """
    from scipy.stats import binomtest, wilcoxon

    a = np.asarray(scores_a, float)
    b = np.asarray(scores_b, float)
    if a.shape != b.shape:
        raise ValueError("配对分数长度必须一致")
    delta = a - b
    better = delta > 0 if higher_is_better else delta < 0
    worse = delta < 0 if higher_is_better else delta > 0
    wins, losses = int(better.sum()), int(worse.sum())
    sign_p = (float(binomtest(wins, wins + losses, 0.5).pvalue)
              if wins + losses else 1.0)
    try:
        wilcoxon_p = float(wilcoxon(a, b).pvalue) if np.any(delta != 0) else 1.0
    except ValueError:
        wilcoxon_p = 1.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float(delta.mean()), "wins": wins, "losses": losses,
            "ties": int(len(a) - wins - losses),
            "sign_test_p": sign_p, "wilcoxon_p": wilcoxon_p,
            "n_pairs": int(len(a))}


def bonferroni_threshold(n_comparisons, alpha=0.05):
    """多重比较阈值；比较前先说明打算做多少次检验"""
    if n_comparisons < 1:
        raise ValueError("比较次数必须为正")
    return alpha / n_comparisons


def d_optimal_designs(X, candidate_rows, size, top_fraction=0.10, kernel=None):
    """按 det(K) 枚举设计并返回最优一族

    设计选择必须允许"独立抽样"才能做统计比较：单一最优设计在固定记录池下
    重复抽样几乎不产生新单元（实测30个种子仅2个不同初始集）。

    参数：
        X: (n, d) 输入矩阵
        candidate_rows: 可供选择的记录索引
        size: 每个设计的点数
        top_fraction: 返回前多少比例的设计作为"D-optimal 家族"
        kernel: (Z1, Z2) -> 协方差的可调用对象；None 时用恒等特征的 Z Z^T

    返回：
        [(combo, det), ...] 按 det 降序
    """
    candidate_rows = [int(r) for r in candidate_rows]
    labels = cell_labels(X[candidate_rows])
    # 每个格取首条记录作为代表，避免同格重复进入设计
    representatives = []
    for cell in np.unique(labels):
        representatives.append(candidate_rows[int(np.flatnonzero(labels == cell)[0])])
    out = []
    for combo in itertools.combinations(representatives, size):
        rows = list(combo)
        Z = np.asarray(X[rows], float)
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 1e-12, Z.std(axis=0), 1.0)
        gram = (Z @ Z.T) if kernel is None else kernel(Z, Z)
        gram = gram + 1e-8 * np.eye(len(rows))
        out.append((combo, float(abs(np.linalg.det(gram)))))
    out.sort(key=lambda pair: -pair[1])
    keep = max(1, int(len(out) * top_fraction))
    return out[:keep]


def design_audit(root=None):
    """对本项目 DRIE 数据做一次设计审计，返回可写入验证记录的字典

    审计项：记录数、格数、重复格分布、按记录随机抽样落到重复格的比例、
    逐目标噪声上限。这些数字决定了后续能否按记录留出、以及 R² 的上限在哪。
    """
    from .data import load_dataset

    from pathlib import Path
    dataset = load_dataset(Path(root) if root else Path(__file__).resolve().parents[2],
                           "real", exclude_ids=(24,))
    X, Y = dataset.X, dataset.Y
    labels = cell_labels(X, DEFAULT_CELL_COLUMNS)
    counts = cell_counts(labels)
    rng = np.random.default_rng(0)

    # 重复格比例取决于抽样池，两个口径都要报：全体记录 vs 主动学习的17条候选池
    al_test_ids = (1, 5, 12, 14, 16, 17)
    pool = np.array([i for i, rid in enumerate(dataset.ids)
                     if int(rid) not in al_test_ids])

    def duplicate_rate(rows, draws=300, size=6):
        hits = 0
        for _ in range(draws):
            picked = rng.choice(rows, size, replace=False)
            if len({int(labels[i]) for i in picked}) < size:
                hits += 1
        return hits / draws

    duplicate_all = duplicate_rate(np.arange(len(X)))
    duplicate_pool = duplicate_rate(pool)
    ceiling = noise_ceiling(Y, labels)
    names = dataset.target_names
    cd = [v for v, n in zip(ceiling["per_target"], names) if "CD" in n]
    ar = [v for v, n in zip(ceiling["per_target"], names) if "AR" in n]
    return {
        "n_records": int(len(X)),
        "distinct_4d_inputs": int(len(np.unique(X, axis=0))),
        "cells": int(len(counts)),
        "cell_sizes": sorted(counts.tolist(), reverse=True),
        "replicated_cells": int((counts > 1).sum()),
        "random_draw_duplicate_cell_rate_all_records": duplicate_all,
        "random_draw_duplicate_cell_rate_al_pool": duplicate_pool,
        "noise_ceiling_macro": ceiling["macro"],
        "noise_ceiling_cd_mean": float(np.mean(cd)) if cd else None,
        "noise_ceiling_ar_mean": float(np.mean(ar)) if ar else None,
        "noise_ceiling_per_target": dict(zip(names, ceiling["per_target"])),
        "recommended_holdout_unit": "cell",
    }


def main():
    """命令行报告：打印本项目的设计审计结果"""
    import json
    print(json.dumps(design_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
