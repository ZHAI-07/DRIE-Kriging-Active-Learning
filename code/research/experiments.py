"""来源阻断的预测检查与配对离线主动学习重放

文件功能：
- basic_evaluation: 留一来源组交叉验证（2/13共同阻断）
- run_active_learning: 配对主动学习重放（30种子×4策略）
- summarize_active_learning: AUC汇总与bootstrap配对检验
- metadata: 运行元数据记录（单位、版本、数学定义）

数学方法：
- 来源阻断交叉验证：每次留出一个来源组（2和13共同处理）
- 固定评估尺度：从初始训练集拟合，主动学习中不更新
- 配对重放：所有策略共享同一分割（消除随机性）
- Bootstrap CI：蒙特卡洛重采样（10000次）估计配对差异置信区间

重现性保证：
- 输入归一化器固定在初始训练集
- 评估尺度固定在初始训练集
- 检查点签名包含数据哈希、代码哈希、配置
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .core import GPConfig, IndependentGP, safe_scale, select_next
from .data import LabelOracle, make_split, source_groups


STRATEGIES = ("random", "uncertainty", "maximin", "duw")


def write_json(path, obj):
    """原子写入JSON（先写临时文件再重命名）"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def clean_json(value):
    """递归清理JSON：numpy类型转Python原生，NaN转None"""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def frame_csv(frame, path):
    """保存DataFrame为CSV（UTF-8 BOM）"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def basic_evaluation(dataset, out, gp_config):
    """留一来源组交叉验证（来源阻断）
    
    验证方式：
    - 每次留出一个来源组作为测试集
    - 2和13共享来源，必须同时留出或同时训练
    - 比较GPR、Ridge、均值基线三种模型
    
    模型：
    - GPR: 本项目的IndependentGP（带不确定性估计）
    - Ridge: 固定α=1的线性回归（无不确定性）
    - Mean: 训练集均值（最简单基线）
    
    参数：
        dataset: Dataset对象
        out: 输出目录
        gp_config: GPConfig对象
    
    返回：
        汇总指标DataFrame（每模型×每响应的RMSE/MAE/R²/覆盖率）
    """
    rows, diagnostics = [], []
    groups = source_groups(dataset)
    for fold, group in enumerate(np.unique(groups)):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        X, Y = dataset.X[train], dataset.Y[train]
        scaler = StandardScaler().fit(X)
        gp = IndependentGP(gp_config, seed=fold).fit(X, Y, x_scaler=scaler)
        gp_mu, gp_sd = gp.predict(dataset.X[test])
        
        # Ridge回归：固定正则化系数α=1（预先声明，不在外层得分上选择）
        ym, ys = Y.mean(axis=0), safe_scale(Y)
        ridge = Ridge(alpha=1.).fit(scaler.transform(X), (Y-ym)/ys)
        ridge_standard = np.asarray(ridge.predict(scaler.transform(dataset.X[test]))).reshape(len(test), Y.shape[1])
        ridge_mu = ridge_standard * ys + ym
        
        # 均值基线
        means = np.tile(ym, (len(test), 1))
        predictions = {"GPR": (gp_mu, gp_sd), "Ridge": (ridge_mu, None),
                       "Mean": (means, None)}
        
        # 记录每个测试样本×每个响应的预测结果
        for model, (mu, std) in predictions.items():
            for k, idx in enumerate(test):
                for j, target in enumerate(dataset.target_names):
                    uncertainty = float(std[k, j]) if std is not None else np.nan
                    rows.append({"model": model, "experiment_id": int(dataset.ids[idx]),
                                 "fold": fold, "source_group": int(group),
                                 "n_train": len(train), "target": target,
                                 "structure": dataset.structures[j], "metric": dataset.metrics[j],
                                 "y_true": dataset.Y[idx, j], "y_pred": mu[k, j],
                                 "std": uncertainty,
                                 "lower95": mu[k, j] - 1.96 * uncertainty,
                                 "upper95": mu[k, j] + 1.96 * uncertainty})
        diagnostics.append({"fold": fold, "train_ids": dataset.ids[train].tolist(),
                            "test_ids": dataset.ids[test].tolist(), **gp.diagnostics()})
    
    # 汇总每个模型×响应的指标
    predictions = pd.DataFrame(rows)
    summary = []
    for (model, target, structure, metric), part in predictions.groupby(
            ["model", "target", "structure", "metric"], sort=False):
        error = part.y_pred.to_numpy() - part.y_true.to_numpy()
        has_interval = part["std"].notna().all()
        summary.append({"model": model, "target": target, "structure": structure,
                        "metric": metric, "n_predictions": len(part),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "mae": float(np.mean(abs(error))),
                        "r2": float(r2_score(part.y_true, part.y_pred)),
                        "coverage95": float(((part.y_true >= part.lower95) &
                                             (part.y_true <= part.upper95)).mean()) if has_interval else np.nan,
                        "mean_interval_width": float((part.upper95-part.lower95).mean()) if has_interval else np.nan})
    frame_csv(predictions, Path(out)/"basic_predictions.csv")
    frame_csv(pd.DataFrame(summary), Path(out)/"basic_metrics.csv")
    write_json(Path(out)/"basic_fit_diagnostics.json", clean_json(diagnostics))
    return pd.DataFrame(summary)


def _metric_rows(dataset, seed, strategy, step, train_size, true, pred, std, eval_scale):
    """计算单步主动学习的评估指标（RMSE/MAE/NRMSE/覆盖率/区间宽度）
    
    eval_scale: 固定在初始训练集的评估尺度（归一化RMSE的分母）
    返回: 逐响应指标行 + 宏平均行
    """
    rmse = np.sqrt(np.mean((pred - true)**2, axis=0))
    mae = np.mean(abs(pred - true), axis=0)
    nr = rmse/eval_scale  # 归一化RMSE
    coverage = np.mean(abs(pred-true) <= 1.96*std, axis=0)  # 95%区间覆盖率
    widths = (3.92*std).mean(axis=0)  # 平均区间宽度（±1.96σ）
    common = {"seed": seed, "strategy": strategy, "step": step, "n_train": train_size}
    
    # 逐响应指标
    rows = [{"target": target, "structure": dataset.structures[j],
             "metric": dataset.metrics[j], "rmse": rmse[j], "mae": mae[j],
             "nrmse": nr[j], "coverage95": coverage[j],
             "mean_interval_width": widths[j], **common} for j, target in enumerate(dataset.target_names)]
    
    # 宏平均：所有响应归一化后取均值
    rows.append({"target": "__macro__", "structure": "all",
                 "metric": "normalized", "rmse": nr.mean(),
                 "mae": (mae/eval_scale).mean(), "nrmse": nr.mean(),
                 "coverage95": coverage.mean(),
                 "mean_interval_width": (widths/eval_scale).mean(), **common})
    
    # 按指标分组的宏平均（CD Loss和AR分别统计）
    for metric, short in (("CD Loss", "cd"), ("AR", "ar")):
        mask = np.array(dataset.metrics) == metric
        if mask.any():
            rows.append({"target": f"__macro_{short}__", "structure": "all",
                         "metric": metric, "rmse": rmse[mask].mean(),
                         "mae": mae[mask].mean(), "nrmse": nr[mask].mean(),
                         "coverage95": coverage[mask].mean(),
                         "mean_interval_width": widths[mask].mean(), **common})
    return rows


def _seed_job(dataset, seed, experiment_config, gp_dict, out, signature):
    """单种子主动学习任务（支持并行执行）
    
    流程：
    1. 检查检查点（签名匹配则复用，不匹配则报错）
    2. 生成分割（初始/候选/测试）
    3. 固定归一化器和评估尺度
    4. 四种策略配对重放（共享分割和GP缓存）
    5. 保存检查点和诊断
    
    签名：数据哈希+配置哈希+代码哈希，确保可重现性
    """
    started = time.perf_counter()
    gp_config = GPConfig(**gp_dict)
    checkpoint = Path(out)/"checkpoints"/f"seed_{seed}.json"
    
    # 检查点复用：签名匹配则直接返回，不匹配则报错
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return previous
        raise RuntimeError(f"Checkpoint configuration changed: {checkpoint}")
    
    # 生成训练/候选/测试分割
    split = make_split(dataset, seed, experiment_config["n_test"],
                       experiment_config["n_initial"], experiment_config["budget"])
    initial, pool, test = split["initial"], split["pool"], split["test"]
    
    # 固定归一化器和评估尺度（仅从初始训练集拟合）
    scaler = StandardScaler().fit(dataset.X[initial])
    scaled_X = scaler.transform(dataset.X)
    eval_scale = safe_scale(dataset.Y[initial])
    reference_scale = float(pdist(scaled_X[np.r_[initial, pool]]).max())  # 输入空间最大距离
    
    cache, diagnostics = {}, []  # GP缓存（避免重复训练）
    curves, selections = [], []
    
    # 四种策略配对重放
    for strategy in STRATEGIES:
        oracle = LabelOracle(dataset, initial, pool, test)
        rng = np.random.default_rng(seed + 918273)
        
        # 主动学习循环（budget轮采样 + 初始评估）
        for step in range(experiment_config["budget"] + 1):
            train_idx, train_X, train_Y = oracle.training()
            key = tuple(train_idx.tolist())
            
            # GP缓存：相同训练集只训练一次
            if key not in cache:
                gp = IndependentGP(gp_config, seed=seed).fit(train_X, train_Y, x_scaler=scaler)
                pred, std = gp.predict(dataset.X)
                cache[key] = (pred, std, gp.y_scale.copy())
                diagnostics.append({"training_ids": dataset.ids[train_idx].tolist(), **gp.diagnostics()})
            pred, std, current_y_scale = cache[key]
            
            # 测试集评估（标签仅用于评分，不进入训练或选点）
            curves.extend(_metric_rows(dataset, seed, strategy, step, len(train_idx),
                                       dataset.Y[test], pred[test], std[test], eval_scale))
            
            if step == experiment_config["budget"]:
                break  # 已达预算，停止采样
            
            # 选择下一个候选点
            candidates = np.array(sorted(oracle.candidates), dtype=int)
            chosen, scores = select_next(
                strategy, scaled_X[candidates], scaled_X[train_idx],
                std[candidates]/current_y_scale, step, experiment_config["budget"], rng,
                base=experiment_config["duw_base"], decay=experiment_config["duw_decay"],
                penalty_mode=experiment_config["penalty_mode"], reference_scale=reference_scale)
            idx = int(candidates[chosen])
            selections.append({"seed": seed, "strategy": strategy,
                               "acquisition_step": step + 1,
                               "experiment_id": int(dataset.ids[idx]),
                               "score": float(scores[chosen]) if scores is not None else None,
                               "n_train_before": len(train_idx),
                               "remaining_candidates_before": len(candidates)})
            oracle.reveal(idx)  # 揭示标签，加入训练集
    
    # 构建检查点记录
    split_record = {"seed": seed, "test_ids": dataset.ids[test].tolist(),
                    "initial_ids": dataset.ids[initial].tolist(),
                    "pool_ids": dataset.ids[pool].tolist(),
                    "rejection_attempts": split["rejection_attempts"],
                    "initial_evaluation_scale": eval_scale.tolist(),
                    "fixed_input_mean": scaler.mean_.tolist(),
                    "fixed_input_scale": scaler.scale_.tolist(),
                    "flagged_id_locations": {str(flag): next((name for name, idx in
                        [("test", test), ("initial", initial), ("pool", pool)]
                        if flag in dataset.ids[idx]), "excluded_or_not_applicable")
                        for flag in (2, 13, 24)}}
    result = clean_json({"signature": signature, "seed": seed, "split": split_record,
                         "curves": curves, "selections": selections,
                         "fit_count": len(diagnostics), "elapsed_seconds": time.perf_counter()-started})
    write_json(Path(out)/"fit_diagnostics"/f"seed_{seed}.json", clean_json(diagnostics))
    write_json(checkpoint, result)
    return result


def summarize_active_learning(curves, out, bootstrap_repeats=10000):
    """主动学习结果汇总与bootstrap配对检验
    
    计算：
    - AUC: 梯形积分RMSE×step（越小越好，非ROC-AUC）
    - 配对差异：每个种子上策略vs random的改进
    - Bootstrap CI: 蒙特卡洛重采样估计置信区间
    
    范围：固定数据集上的随机分割Monte Carlo，非独立物理实验
    """
    auc_rows = []
    for key, part in curves.groupby(["seed", "strategy", "target", "structure", "metric"], sort=False):
        part = part.sort_values("step")
        auc = float(np.trapezoid(part.rmse.to_numpy(), x=part.step.to_numpy()))
        n_auc = float(np.trapezoid(part.nrmse.to_numpy(), x=part.step.to_numpy()))
        span = float(part.step.max()-part.step.min())
        seed, strategy, target, structure, metric = key
        auc_rows.append({"seed": seed, "strategy": strategy, "target": target,
                         "structure": structure, "metric": metric, "test_auc": auc,
                         "normalized_test_auc": n_auc,
                         "mean_curve_rmse": auc/span if span else float(part.rmse.iloc[-1]),
                         "final_rmse": float(part.rmse.iloc[-1]),
                         "final_nrmse": float(part.nrmse.iloc[-1])})
    auc = pd.DataFrame(auc_rows)
    summary = auc.groupby(["strategy", "target", "structure", "metric"], sort=False).agg(
        n_repeats=("seed", "count"), mean_test_auc=("test_auc", "mean"),
        sd_test_auc=("test_auc", "std"), mean_final_rmse=("final_rmse", "mean"),
        sd_final_rmse=("final_rmse", "std"), mean_final_nrmse=("final_nrmse", "mean")).reset_index()
    paired = []
    rng = np.random.default_rng(20261109)
    for (target, structure, metric), part in auc.groupby(["target", "structure", "metric"], sort=False):
        table = part.pivot(index="seed", columns="strategy", values="test_auc")
        for strategy in ("uncertainty", "maximin", "duw"):
            pair = table[["random", strategy]].dropna()
            baseline = pair["random"].to_numpy()
            differences = pair[strategy].to_numpy() - baseline
            n = len(pair)
            draws = rng.integers(0, n, size=(bootstrap_repeats, n))
            low, high = np.quantile(differences[draws].mean(axis=1), [.025, .975])
            relative = np.divide(-differences, baseline, out=np.full(n, np.nan), where=abs(baseline)>1e-12)*100
            paired.append({"strategy": strategy, "baseline": "random", "target": target,
                           "structure": structure, "metric": metric, "n_paired_seeds": n,
                           "mean_auc_difference": differences.mean(),
                           "mc_bootstrap_ci95_low": low, "mc_bootstrap_ci95_high": high,
                           "mean_relative_improvement_percent": np.nanmean(relative),
                           "fraction_seeds_better": np.mean(differences < 0),
                           "interval_scope": "Monte Carlo split/selection variability conditional on this fixed dataset; not independent physical experiments"})
    frame_csv(auc, Path(out)/"al_auc.csv")
    frame_csv(summary, Path(out)/"al_summary.csv")
    frame_csv(pd.DataFrame(paired), Path(out)/"al_paired.csv")
    return summary


def run_active_learning(dataset, out, experiment_config, gp_config, workers=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"data_hash": dataset.source_sha256, "ids": dataset.ids.tolist(),
                "numerical_values_sha256": hashlib.sha256(dataset.Y.tobytes()).hexdigest(),
                "experiment": experiment_config, "gp": asdict(gp_config),
                "method_version": "2026-11-09-v1-source-blocked-independent-gp",
                "core_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob("*.py")}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    seeds = list(range(experiment_config["seed_start"],
                       experiment_config["seed_start"]+experiment_config["repeats"]))
    results = []
    if workers == 1:
        for seed in seeds:
            result = _seed_job(dataset, seed, experiment_config, asdict(gp_config), out, signature)
            results.append(result)
            print(f"seed {seed}: {result['fit_count']} distinct fits, {result['elapsed_seconds']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(4, workers)) as executor:
            futures = {executor.submit(_seed_job, dataset, seed, experiment_config,
                                        asdict(gp_config), out, signature): seed for seed in seeds}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"seed {result['seed']}: {result['fit_count']} distinct fits, {result['elapsed_seconds']:.1f}s", flush=True)
    results.sort(key=lambda r: r["seed"])
    curves = pd.DataFrame([row for result in results for row in result["curves"]])
    selections = pd.DataFrame([row for result in results for row in result["selections"]])
    # Exact checks on physical experimental budgets and paired partitions.
    expected = experiment_config["budget"]
    if not selections.groupby(["seed", "strategy"]).size().eq(expected).all():
        raise RuntimeError("Unequal acquisition budgets")
    if not selections.groupby(["seed", "strategy"]).experiment_id.nunique().eq(expected).all():
        raise RuntimeError("An experiment was acquired twice")
    frame_csv(curves, out/"al_curves.csv")
    frame_csv(selections, out/"al_selections.csv")
    write_json(out/"al_splits.json", [r["split"] for r in results])
    summary = summarize_active_learning(curves, out)
    write_json(out/"al_execution.json", {"signature": signature,
        "seeds": seeds, "fits_per_seed": [r["fit_count"] for r in results],
        "seconds_per_seed": [r["elapsed_seconds"] for r in results],
        "completed_repeats": len(results), "strategies": list(STRATEGIES)})
    return summary


def metadata(dataset, config, gp_config):
    return {"dataset": dataset.name, "source": dataset.source_path,
            "source_sha256": dataset.source_sha256, "source_provenance": dataset.provenance,
            "numerical_values_sha256": hashlib.sha256(dataset.Y.tobytes()).hexdigest(),
            "units": ({"features": ["W", "W", "sccm", "mTorr"],
                       "CD Loss": "micrometre", "AR": "dimensionless",
                       "source": "User confirmed units on 2026-11-09; input snapshot remains unchanged"}
                      if dataset.name == "real" else
                      {"features": ["mT (workbook label)", "W"],
                       "response": "legacy simulation unit; historical code says nm/min, primary provenance not verified"}),
            "experiment_ids": dataset.ids.tolist(), "X_shape": list(dataset.X.shape),
            "Y_shape": list(dataset.Y.shape), "target_names": dataset.target_names,
            "configuration": config, "gp_configuration": asdict(gp_config),
            "versions": {"python": sys.version, "platform": platform.platform(),
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "evaluation_unit": "process experiment; 2/13 kept on the same train/test side because of shared measurement-source evidence",
            "acquisition_unit": "one recipe ID reveals all available target responses at once",
            "duw_reproduction_scope": "legacy weighted score exactly; multiresponse standardized-SD averaging and common query sequence are this project's adaptation",
            "test_auc_definition": "trapezoidal integral of held-out RMSE over acquisition step, includes step 0; lower is better; not ROC-AUC",
            "macro_scale": "fixed per-target SD from each repeat's initial training labels; never recomputed from test or pool labels",
            "fit_scale": "current revealed training labels only; input scaler fixed at initial training values during AL",
            "noise": "WhiteKernel observation-predictive uncertainty plus fixed alpha training diagonal regularization; intervals are model-conditional",
            "source_duplicate_caveat": "2/13 blocked across evaluation sides, but counted as separate recipe IDs in the initial/pool acquisition budget; exclude-each sensitivity needed",
            "statistical_scope": "paired random-split Monte Carlo on one finite dataset, not independent physical replication",
            "status": "running"}
