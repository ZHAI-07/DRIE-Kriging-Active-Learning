"""保守的优化就绪性检查与实测Pareto前沿

文件功能：
- nondominated: 计算非支配解（Pareto前沿）
- optimization_readiness: 保守的工程优化就绪性筛查

数学方法：
- Pareto支配：同时最小化6个CD Loss，最大化6个AR
- 筛查条件：每个响应GPR RMSE优于均值基线，R²>0，95%覆盖率≥0.75

工程决策：
- 仅报告实测Pareto备选方案（24个实验中的子集）
- 不制造连续域优化配方（缺乏新物理验证实验）
- CFD数据集跳过优化环节（算法验证任务）
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .experiments import frame_csv, write_json, clean_json


def nondominated(costs):
    """计算非支配解（Pareto前沿）
    
    定义：所有列均为最小化目标，i支配j当且仅当：
    - costs[i] <= costs[j] 在所有维度
    - 至少一个维度严格小于
    
    参数：
        costs: (n_samples, n_objectives) 目标值矩阵
    
    返回：
        布尔掩码，True表示非支配解
    """
    costs = np.asarray(costs, float)
    mask = np.ones(len(costs), dtype=bool)
    for i in range(len(costs)):
        # i被j支配 <=> 存在j使得j<=i且j<i在至少一维
        mask[i] = not np.any(np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1))
    return mask


def optimization_readiness(dataset, out, basic_metrics):
    """工程优化就绪性检查与Pareto前沿导出
    
    流程：
    1. CFD数据集直接跳过（算法验证，不产生DRIE配方）
    2. 计算Pareto前沿（AR取负号转为最小化）
    3. 保守筛查：每个响应GPR优于均值基线、R²>0、覆盖率≥75%
    4. 记录决策：不支持连续配方推荐（数据质量标记未解决）
    
    参数：
        dataset: Dataset对象
        out: 输出目录
        basic_metrics: basic_evaluation返回的预测指标DataFrame
    
    输出：
        optimization_status.json - 决策记录
        observed_pareto.csv - 实测Pareto点（仅DRIE）
    """
    out = Path(out)
    if dataset.name != "real":
        write_json(out/"optimization_status.json", {
            "status": "not_targeted", "reason": "CFD is a separate algorithm-validation task; it does not supply DRIE recipe recommendations."})
        return
    
    # 构建最小化成本矩阵（AR取负号）
    costs = dataset.Y.copy()
    for j, metric in enumerate(dataset.metrics):
        if metric == "AR":
            costs[:, j] *= -1
    
    # 计算非支配解
    pareto = nondominated(costs)
    
    # 导出包含Pareto标记的实测数据
    observed = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    observed.insert(0, "experiment_id", dataset.ids)
    observed["pareto_all_12_responses"] = pareto
    for j, target in enumerate(dataset.target_names):
        observed[target] = dataset.Y[:, j]
    frame_csv(observed, out/"observed_pareto.csv")
    
    # 保守预测质量筛查（仅当有basic_metrics时）
    checks = []
    if basic_metrics is not None:
        for target in dataset.target_names:
            p = basic_metrics[basic_metrics.target == target].set_index("model")
            gpr, baseline = p.loc["GPR"], p.loc["Mean"]
            checks.append({"target": target, "gpr_rmse": gpr.rmse,
                           "mean_rmse": baseline.rmse, "gpr_r2": gpr.r2,
                           "coverage95": gpr.coverage95,
                           "beats_mean": bool(gpr.rmse < baseline.rmse),
                           "positive_r2": bool(gpr.r2 > 0),
                           "coverage_screen": bool(gpr.coverage95 >= .75)})
    
    # 所有响应都通过筛查才标记就绪
    ready = bool(checks) and all(c["beats_mean"] and c["positive_r2"] and c["coverage_screen"] for c in checks)
    
    # 工程决策记录
    result = {"status": "continuous_recipe_recommendation_not_supported",
              "predictive_screen_passed": ready, "response_checks": checks,
              "screen_definition": "Predeclared conservative screen: every response beats its training-mean baseline, has pooled fold-out R2 > 0, and empirical 95% interval coverage >= 0.75. This heuristic screen is not proof of engineering validity.",
              "observed_pareto_ids": dataset.ids[pareto].tolist(),
              "pareto_definition": "Minimize all six CD Loss and maximize all six AR simultaneously; no undeclared averaging or weights.",
              "constraints": ["Raw-response source questions for experiments 2, 13 and 24 remain unresolved unless excluded in a named sensitivity run.",
                              "Equipment increments, fixed cycle/time settings and per-structure engineering acceptance limits need confirmation; four input units were user-confirmed.",
                              "No new physical etching experiment is available for verification."],
              "decision": "Report measured Pareto alternatives and prediction diagnostics. Do not manufacture a continuously optimized recipe or claim physical improvement.",
              "scope": "Measured alternatives in this finite table; not a continuous-domain optimum. Data-quality flags also apply to observed Pareto status."}
    write_json(out/"optimization_status.json", clean_json(result))
