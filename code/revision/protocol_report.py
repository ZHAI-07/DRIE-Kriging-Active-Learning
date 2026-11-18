"""把评估协议口径加到已有模型对照结果上（纯读，不重算）

本脚本不拟合任何模型，只读取 `outputs/revision/models/<scenario>/predictions.csv`
与 `metrics.csv`，按 `research.protocol` 的口径补充三项信息：

1. 设计审计：记录数、设计格数、重复格分布；
2. 噪声上限：每个目标的 `1 - SS_within/SS_total`，以及与已保存 R² 的比值，
   即"占可达上限的比例"——脱离上限谈 R² 高低没有意义；
3. 格泄漏暴露率：已保存的预测按**来源组**留出（即按记录留出），被留出记录的
   同格重复样本可能仍在训练集。本脚本统计受此影响的预测占比，给出已保存
   数字被抬高的直接度量。

输出为新文件，不覆盖任何既有验收记录：
    outputs/revision/models/protocol_report.json
    outputs/revision/models/protocol_report.csv

用法：python -B code/revision/protocol_report.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))

from research.data import load_dataset, source_groups
from research.protocol import (cell_labels, cell_table, noise_ceiling,
                               pooled_metrics)

MODELS = ROOT / "outputs/revision/models"
EXCLUDE = {"real_all24": (), "real_without24": (24,), "real_without2_24": (2, 24),
           "real_without13_24": (13, 24)}
# 只处理 DRIE 真实数据的四个情景；cfd_* 为单目标仿真，口径不同
SCENARIOS = tuple(EXCLUDE)


def scenario_audit(name):
    """按该情景实际使用的记录做设计审计与噪声上限"""
    data = load_dataset(ROOT, "real", exclude_ids=EXCLUDE[name])
    labels = cell_labels(data.X)
    ceiling = noise_ceiling(data.Y, labels)
    counts = np.bincount(labels)
    return data, labels, {
        "n_records": int(len(data.X)),
        "cells": int(len(counts)),
        "replicated_cells": int((counts > 1).sum()),
        "cell_sizes": sorted(counts.tolist(), reverse=True),
        "noise_ceiling_macro": ceiling["macro"],
        "noise_ceiling_per_target": dict(zip(data.target_names,
                                             ceiling["per_target"])),
    }


def cell_leakage(predictions, data, labels):
    """统计被留出记录中，同格其它记录仍留在训练集的比例

    已保存的预测按来源组留出，等价于按记录留出。若某记录的同格兄弟不在同一
    来源组，则该预测的目标点输入在训练集里已出现，属于泄漏。用 (fold,
    source_group) 反推训练集，再按格比对。
    """
    groups = source_groups(data)
    id_to_row = {int(rid): i for i, rid in enumerate(data.ids)}
    per_fold_train = {}
    hits, total = 0, 0
    for fold, part in predictions.groupby("fold"):
        group = int(part["source_group"].iloc[0])
        train_rows = np.flatnonzero(groups != group)
        cells_in_train = set(int(labels[i]) for i in train_rows)
        per_fold_train[int(fold)] = len(cells_in_train)
        rows = [id_to_row[int(exp)] for exp in part["experiment_id"].unique()]
        for row in rows:
            for _ in range(predictions[(predictions.fold == fold) &
                                       (predictions.experiment_id == data.ids[row])].shape[0]):
                total += 1
                hits += int(labels[row]) in cells_in_train
    return hits / total if total else None, total


def exclusion_effect():
    """记录24号排除前后的噪声上限对比

    24号位于(920,40)格，与8、22号同格，但AR=34.133而两者为20.863/20.887，
    偏离格均值8.84（全体AR的sd为2.668）。它把该格格内方差抬高到使AR上限从
    0.868跌到0.311，全体上限从0.884跌到0.586。因此排除24号不只是"来源不同"
    的取舍，而是可达精度上界的直接决定因素。
    """
    out = {}
    for label, exclude in (("with_24", ()), ("without_24", (24,))):
        data = load_dataset(ROOT, "real", exclude_ids=exclude)
        ceiling = noise_ceiling(data.Y, cell_labels(data.X))
        names = data.target_names
        cd = [v for v, n in zip(ceiling["per_target"], names) if "CD" in n]
        ar = [v for v, n in zip(ceiling["per_target"], names) if "AR" in n]
        out[label] = {"n_records": int(len(data.X)),
                      "ceiling_cd_mean": float(np.mean(cd)),
                      "ceiling_ar_mean": float(np.mean(ar)),
                      "ceiling_macro": ceiling["macro"]}
    data = load_dataset(ROOT, "real")
    rows = [int(np.flatnonzero(data.ids == rid)[0]) for rid in (8, 22, 24)]
    ar_values = [float(data.Y[r, 1]) for r in rows]
    out["record_24_detail"] = {
        "cell": [920.0, 40.0], "cell_mate_ids": [8, 22, 24],
        "ar_values": ar_values,
        "deviation_from_cell_mean": ar_values[2] - float(np.mean(ar_values)),
        "deviation_in_global_sd": (ar_values[2] - float(np.mean(ar_values)))
                                  / float(data.Y[:, 1].std()),
    }
    return out


def main():
    rows, report = [], {"holdout_unit_of_saved_results": "record (source group)",
                        "note": "saved predictions hold out source groups, which "
                                "leaves same-cell replicates in training",
                        "record_exclusion_effect": exclusion_effect(),
                        "scenarios": {}}
    for name in SCENARIOS:
        folder = MODELS / name
        predictions_path = folder / "predictions.csv"
        metrics_path = folder / "metrics.csv"
        if not predictions_path.exists() or not metrics_path.exists():
            continue
        data, labels, audit = scenario_audit(name)
        predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
        metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
        leak_rate, n_pred = cell_leakage(predictions, data, labels)
        ceiling = audit["noise_ceiling_per_target"]

        per_model = {}
        for model, part in metrics.groupby("model"):
            target_rows = part.set_index("target")
            present = [t for t in ceiling if t in target_rows.index]
            if not present:
                continue
            r2 = np.array([float(target_rows.loc[t, "r2"]) for t in present])
            cap = np.array([ceiling[t] for t in present])
            per_model[model] = {
                "macro_r2": float(np.mean(r2)),
                "macro_noise_ceiling": float(np.mean(cap)),
                "r2_fraction_of_ceiling": float(np.mean(r2 / cap)),
                "n_targets": len(present),
            }
            rows.append({"scenario": name, "model": model,
                         "macro_r2": float(np.mean(r2)),
                         "noise_ceiling": float(np.mean(cap)),
                         "r2_fraction_of_ceiling": float(np.mean(r2 / cap)),
                         "cell_leakage_exposure": leak_rate})
        report["scenarios"][name] = {**audit, "cell_leakage_exposure": leak_rate,
                                     "n_predictions_checked": n_pred,
                                     "models": per_model}

    frame = pd.DataFrame(rows)
    frame.to_csv(MODELS / "protocol_report.csv", index=False)
    (MODELS / "protocol_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, block in report["scenarios"].items():
        print(f"\n== {name} ==  {block['n_records']} 条 / {block['cells']} 格 "
              f"({block['replicated_cells']} 格有重复)  "
              f"格泄漏暴露率 {block['cell_leakage_exposure']:.1%}")
        print(f"   噪声上限 {block['noise_ceiling_macro']:.3f}")
        for model, m in sorted(block["models"].items(),
                               key=lambda kv: -kv[1]["macro_r2"]):
            print(f"   {model:26s} R² {m['macro_r2']:+.3f}   "
                  f"占上限 {m['r2_fraction_of_ceiling']:+.1%}")
    print(f"\n写入 {MODELS / 'protocol_report.json'}")
    return report


if __name__ == "__main__":
    main()
