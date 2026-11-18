"""固定核高斯过程：本项目选定的建模方案

为什么是这个模型
----------------
在小样本（23 条记录、15 个设计格）条件下的实测结果：

* 4 维满超参 GP（ARD Matern52 + 白噪声，逐折优化）按记录留出 R²=-0.175，
  按格留出 -0.094：优化 4 维核的超参在 n≈22 时是插值，不是学习；
* 逐折 MLE 调参的 2D GP 按格留出 +0.062，也明显不如固定核的 +0.209；
* 内层 CV 与外层表现的 Spearman 为 **-0.362**，即在这个样本量下"调参选优"
  与真实性能反向相关，所以核超参必须固定，不能每折重选；
* 因此采用在本数据上固定的一组超参：Matern 5/2、长度尺度 1.2（标准化坐标）、
  信号 2.5、观测噪声 0.3，输入取 Coil + Platen。

数学约定
--------
    k(x,x') = signal * f(s),   s = sqrt(5) * ||(x-x')/tau_dim||
    f(s) = (1 + s + s^2/3) exp(-s)
    K = k(Z,Z) + (noise + jitter) I            # 对角线含观测噪声
    预测行使用无噪声核：mu = k(Z_test,Z) K^-1 z
    预测方差：(signal + noise) - k(Z_test,Z) K^-1 k(Z,Z_test)

标准化：X 与 y 都用"冻结 scaler"（由参考集算出后固定），全共形要求拟合过程
对合并样本对称，若按候选标签重新标准化 y，分数会依赖候选值本身。实测：按增广
集标准化会产生 ±2000 的区间（见 _tmp 探索记录），冻结后覆盖率为 0.955。

全共形
------
rank = ceil((n+1)(1-alpha)) 需 <= n 才有限，即 n >= 1/alpha - 1（95% 要 19 点，
90% 要 9 点）。n_train<=14 时 95% 必然退化为整条实轴，这条已在
formal/Formal/Conformal.lean 中机器检查（conformal_rank_exceeds_n）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# 默认超参：在本数据上固定，不做逐折优化（理由见模块文档）
TAU = 1.2
SIGNAL = 2.5
NOISE = 0.3
JITTER = 1e-8
ALPHA_DEFAULT = 0.10


def matern52(s):
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


class FrozenScaler:
    """在参考集上算出后固定的标准化器"""

    def __init__(self, X_ref, y_ref=None):
        X_ref = np.asarray(X_ref, float)
        self.x_mu = X_ref.mean(axis=0)
        sd = X_ref.std(axis=0, ddof=0)
        self.x_sd = np.where(sd > 1e-12, sd, 1.0)
        if y_ref is None:
            self.y_mu, self.y_sd = 0.0, 1.0
        else:
            y_ref = np.asarray(y_ref, float)
            self.y_mu = float(y_ref.mean())
            self.y_sd = float(y_ref.std(ddof=0)) or 1.0

    def x(self, X):
        return (np.asarray(X, float) - self.x_mu) / self.x_sd

    def y(self, y):
        return (np.asarray(y, float) - self.y_mu) / self.y_sd

    def y_inv(self, z):
        return np.asarray(z, float) * self.y_sd + self.y_mu


class FixedKernelGP:
    """固定超参的 Matern 5/2 GP，支持任意输入维度与逐维长度尺度"""

    def __init__(self, scaler, tau=TAU, signal=SIGNAL, noise=NOISE):
        self.scaler = scaler
        self.tau = np.atleast_1d(np.asarray(tau, float))
        self.signal = float(signal)
        self.noise = float(noise)

    # --- 核 ---------------------------------------------------------
    def _tau_vec(self, d):
        return self.tau if self.tau.size == d else np.full(d, float(self.tau[0]))

    def gram(self, A, B):
        tau = self._tau_vec(A.shape[1])
        diff = (A[:, None, :] - B[None, :, :]) / tau[None, None, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        return self.signal * matern52(np.sqrt(5.0) * d)

    def gram_train(self, Z):
        return self.gram(Z, Z) + (self.noise + JITTER) * np.eye(len(Z))

    # --- 拟合与预测 --------------------------------------------------
    def fit(self, X_train, y_train):
        Z = self.scaler.x(X_train)
        self._Z = Z
        self._z = self.scaler.y(y_train)
        self._K = self.gram_train(Z)
        self._chol = cho_factor(self._K, lower=True)
        self._alpha = cho_solve(self._chol, self._z)
        return self

    def predict(self, X_test, return_sd=False):
        Zt = self.scaler.x(X_test)
        ks = self.gram(Zt, self._Z)
        mu = ks @ self._alpha
        if not return_sd:
            return self.scaler.y_inv(mu)
        # 预测方差含观测噪声项，与 sklearn 的 return_std 口径一致
        v = cho_solve(self._chol, ks.T)
        var = (self.signal + self.noise) - np.einsum("ij,ji->i", ks, v)
        return (self.scaler.y_inv(mu),
                np.sqrt(np.maximum(var, 1e-12)) * self.scaler.y_sd)

    # --- 全共形 ------------------------------------------------------
    def conformal_interval(self, y_train, x_test, alpha=ALPHA_DEFAULT,
                           use_cpp=True):
        """单点全共形区间；C++ 加速器可用时走 C++，否则用 numpy 参考实现"""
        mu, sd = self.predict(np.asarray(x_test).reshape(1, -1), return_sd=True)
        center = float(self.scaler.y(mu[0]))
        spread = max(float(sd[0] / self.scaler.y_sd), 1e-6)
        z_test = self.scaler.x(np.asarray(x_test).reshape(1, -1))[0]

        def _numpy():
            from tools.fast_conformal import numpy_interval_std
            return numpy_interval_std(self._Z, self._z, z_test,
                                     self._tau_vec(self._Z.shape[1]),
                                     self.signal, self.noise + JITTER,
                                     center, spread, alpha=alpha)

        if use_cpp:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
                from fast_conformal import interval_std
                lo, hi = interval_std(self._Z, self._z, z_test,
                                      self._tau_vec(self._Z.shape[1]),
                                      self.signal, self.noise + JITTER,
                                      center, spread, alpha=alpha)
                if np.isfinite(lo) and np.isfinite(hi):
                    return (float(self.scaler.y_inv(lo)),
                            float(self.scaler.y_inv(hi)))
            except Exception:
                pass
        lo, hi = _numpy()
        return float(self.scaler.y_inv(lo)), float(self.scaler.y_inv(hi))


def conformal_rank(n, alpha):
    """全共形象；> n 表示该置信水平在本样本量下不可达"""
    return int(math.ceil((n + 1) * (1.0 - alpha)))


def conformal_feasible(n, alpha):
    """该 (n, alpha) 是否存在有限区间"""
    return conformal_rank(n, alpha) <= n
