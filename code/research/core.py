"""高斯过程回归、采集函数与主动学习核心模块

文件功能：
- GPConfig: 高斯过程超参数配置（核函数、噪声边界、优化器参数）
- IndependentGP: 独立多输出GP（每个响应独立训练，手动归一化）
- acquisition_scores: 采集函数（uncertainty/maximin/DUW）
- select_next: 下一个采样点选择

数学方法：
- Kernel: Matérn(ν=2.5) 或 RBF，带ConstantKernel和WhiteKernel
- 归一化: 手动y标准化（population std），输入StandardScaler
- DUW: 加权不确定性与最小距离，权重随迭代衰减
- 优化器: L-BFGS-B，自定义包装记录收敛信息
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GPConfig:
    """高斯过程超参数配置
    
    kernel: "matern52" (Matérn ν=2.5) 或 "rbf" (径向基)
    alpha: 训练对角正则化项（独立于WhiteKernel）
    noise_initial/bounds: WhiteKernel初值与优化边界
    length_bounds: 长度尺度优化边界
    constant_bounds: ConstantKernel优化边界
    restarts: 优化器随机重启次数
    maxiter: L-BFGS-B最大迭代数
    optimize: 是否优化超参数（False则使用初值）
    """
    kernel: str = "matern52"
    alpha: float = 1e-3
    noise_initial: float = 1e-3
    noise_bounds: tuple = (1e-6, 1e-1)
    length_bounds: tuple = (1e-2, 10.0)
    constant_bounds: tuple = (1e-3, 1e3)
    restarts: int = 0
    maxiter: int = 60
    optimize: bool = True


def safe_scale(y):
    """计算训练标签的总体标准差（带常数保护）
    
    使用ddof=0（总体std而非样本std），避免除以过小的尺度
    当std < 1e-12时返回1.0防止数值问题
    """
    y = np.asarray(y, float)
    scale = y.std(axis=0, ddof=0)
    return np.where(scale > 1e-12, scale, 1.0)


class IndependentGP:
    """独立多输出高斯过程回归器
    
    每个响应变量独立训练一个标量GP，输入共享（DRIE: 4维工艺参数）
    
    架构设计：
    - 手动目标归一化使单位和采集函数聚合显式化
    - WhiteKernel贡献观测预测不确定性
    - alpha是额外的训练对角项，遵循遗留惯例
    
    与sklearn.multioutput的区别：
    - 输入scaler固定（主动学习中不随新数据更新）
    - 输出归一化手动控制（便于跨响应聚合）
    """

    def __init__(self, config=GPConfig(), seed=0):
        self.config, self.seed = config, int(seed)

    def _optimizer(self, objective, initial_theta, bounds):
        """自定义优化器包装，记录L-BFGS-B收敛诊断"""
        result = minimize(objective, initial_theta, method="L-BFGS-B", jac=True,
                          bounds=bounds, options={"maxiter": self.config.maxiter,
                                                   "ftol": 1e-9, "gtol": 1e-5})
        self.optimizer_runs.append({"success": bool(result.success),
                                    "status": int(result.status),
                                    "iterations": int(result.nit),
                                    "message": str(result.message)})
        return result.x, float(result.fun)

    def fit(self, X, Y, x_scaler=None):
        """训练多输出GP模型
        
        参数：
            X: 输入特征 (n_samples, n_features)
            Y: 目标响应 (n_samples, n_targets)，可以是1D
            x_scaler: 预拟合的StandardScaler（主动学习中固定在初始训练集）
        
        返回：self（链式调用）
        
        重要：候选池和测试集标签不进入此函数
        """
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if len(X) != len(Y) or not np.isfinite(Y).all():
            raise ValueError("Invalid training data")
        
        # 固定输入归一化器（主动学习中不随新标签更新）
        self.x_scaler = x_scaler if x_scaler is not None else StandardScaler().fit(X)
        # 手动目标归一化：记录均值和尺度
        self.y_mean = Y.mean(axis=0)
        self.y_scale = safe_scale(Y)
        Z = self.x_scaler.transform(X)
        T = (Y - self.y_mean) / self.y_scale  # 标准化目标
        
        self.models, self.fit_warnings, self.optimizer_runs = [], [], []
        # 为每个响应变量独立训练GP
        for j in range(Y.shape[1]):
            # 构建复合核：ConstantKernel * (Matérn或RBF) + WhiteKernel
            if self.config.kernel == "matern52":
                base = Matern(np.ones(X.shape[1]), self.config.length_bounds, nu=2.5)
            elif self.config.kernel == "rbf":
                base = RBF(np.ones(X.shape[1]), self.config.length_bounds)
            else:
                raise ValueError(self.config.kernel)
            kernel = ConstantKernel(1., self.config.constant_bounds) * base
            kernel += WhiteKernel(self.config.noise_initial, self.config.noise_bounds)
            
            # 构建GP回归器（normalize_y=False因为手动归一化了）
            model = GaussianProcessRegressor(
                kernel=kernel, alpha=self.config.alpha, normalize_y=False,
                optimizer=self._optimizer if self.config.optimize else None,
                n_restarts_optimizer=self.config.restarts, random_state=self.seed + j)
            
            # 捕获sklearn的ConvergenceWarning
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(Z, T[:, j])
            self.fit_warnings.extend({"target_index": j, "message": str(w.message)}
                                     for w in caught)
            self.models.append(model)
        return self

    def predict(self, X):
        """预测均值和标准差（原始单位）
        
        参数：
            X: 输入特征 (n_samples, n_features)
        
        返回：
            mu: 预测均值 (n_samples, n_targets)
            sigma: 预测标准差 (n_samples, n_targets)
        """
        Z = self.x_scaler.transform(np.asarray(X, float))
        pairs = [model.predict(Z, return_std=True) for model in self.models]
        mu = np.column_stack([p[0] for p in pairs])
        sigma = np.column_stack([p[1] for p in pairs])
        # 还原到原始单位
        return mu * self.y_scale + self.y_mean, sigma * self.y_scale

    def diagnostics(self):
        """返回训练诊断信息（核参数、归一化尺度、警告等）"""
        return {"kernels": [str(m.kernel_) for m in self.models],
                "theta": [m.kernel_.theta.tolist() for m in self.models],
                "training_target_mean": self.y_mean.tolist(),
                "training_target_scale": self.y_scale.tolist(),
                "input_mean": self.x_scaler.mean_.tolist(),
                "input_scale": self.x_scaler.scale_.tolist(),
                "warnings": self.fit_warnings, "optimizer_runs": self.optimizer_runs}


def unit_range(values):
    """归一化到[0,1]区间（用于采集函数中的值域对齐）
    
    当极差<1e-10时返回全1数组（避免除零）
    """
    values = np.asarray(values, float)
    extent = np.ptp(values)
    return ((values - values.min()) / extent if extent > 1e-10
            else np.ones_like(values))


def acquisition_scores(strategy, X_pool_scaled, X_train_scaled, normalized_std,
                       iteration, budget, base=0.20, decay=0.60,
                       penalty_mode="weighted", reference_scale=1.0):
    """计算采集函数得分（不依赖候选点或测试集标签）
    
    多响应适配：
    - 后验标准差按当前训练目标SD单位平均
    - 一次采集选择一个物理实验，揭示所有输出响应
    
    策略：
    - uncertainty: 纯不确定性采样（后验标准差均值）
    - maximin: 纯探索（到最近训练点的距离）
    - duw: Distance-Uncertainty Weighted（遗留加权公式的精确复现）
    
    DUW加权分支：
    - weighted模式：精确保留遗留minmax/offset/weight公式
    - 权重随迭代衰减：w = base * max(1 - iteration/budget * decay, 0.05)
    
    参数：
        normalized_std: 按训练目标尺度归一化的标准差
        penalty_mode: "weighted"(加权乘积), "hard"(半径内惩罚), "soft"(指数衰减)
        reference_scale: 输入空间的参考尺度（scaled输入的最大距离）
    """
    std = np.asarray(normalized_std, float)
    uncertainty = std.mean(axis=1) if std.ndim == 2 else std  # 多响应时取均值
    distances = cdist(X_pool_scaled, X_train_scaled).min(axis=1)  # 到最近训练点的距离
    
    if strategy == "uncertainty":
        return uncertainty
    if strategy == "maximin":
        return distances
    if strategy != "duw":
        raise ValueError(strategy)
    
    # DUW策略：加权或惩罚模式
    if penalty_mode == "weighted":
        # 遗留加权公式：(u+0.01)^(1-w) * (d+0.01)^w
        w = base * max(1. - iteration / max(budget, 1) * decay, .05)
        return (unit_range(uncertainty) + .01) ** (1. - w) * (unit_range(distances) + .01) ** w
    
    # 自适应半径
    radius = base * max(1. - iteration / max(budget, 1) * decay, .1) * reference_scale
    if penalty_mode == "hard":
        # 硬惩罚：半径内点得分×0.001
        return uncertainty * np.where(distances < radius, .001, 1.)
    if penalty_mode == "soft":
        # 软惩罚：高斯衰减
        return uncertainty * (1. - np.exp(-distances**2 / (2*radius**2 + 1e-10)))
    raise ValueError(penalty_mode)


def select_next(strategy, X_pool_scaled, X_train_scaled, normalized_std,
                iteration, budget, rng, **kwargs):
    """选择下一个采样点
    
    参数：
        strategy: "random", "uncertainty", "maximin", "duw"
        rng: numpy随机数生成器
        **kwargs: 传递给acquisition_scores的额外参数
    
    返回：
        (选中索引, 得分数组) - random策略返回None作为scores
    """
    if strategy == "random":
        return int(rng.integers(len(X_pool_scaled))), None
    scores = acquisition_scores(strategy, X_pool_scaled, X_train_scaled,
                                normalized_std, iteration, budget, **kwargs)
    return int(np.argmax(scores)), scores
