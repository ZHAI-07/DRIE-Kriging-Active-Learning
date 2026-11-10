"""Training-only transforms, GPR models and label-free acquisition functions."""
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
    """Population SD of supplied TRAINING labels, with an explicit constant guard."""
    y = np.asarray(y, float)
    scale = y.std(axis=0, ddof=0)
    return np.where(scale > 1e-12, scale, 1.0)


class IndependentGP:
    """One scalar GP per response; input remains four process features for DRIE.

    Manual target normalization makes units and acquisition aggregation explicit.
    WhiteKernel contributes to returned observation-predictive uncertainty;
    alpha is an additional training diagonal term, matching the legacy convention.
    """

    def __init__(self, config=GPConfig(), seed=0):
        self.config, self.seed = config, int(seed)

    def _optimizer(self, objective, initial_theta, bounds):
        result = minimize(objective, initial_theta, method="L-BFGS-B", jac=True,
                          bounds=bounds, options={"maxiter": self.config.maxiter,
                                                   "ftol": 1e-9, "gtol": 1e-5})
        self.optimizer_runs.append({"success": bool(result.success),
                                    "status": int(result.status),
                                    "iterations": int(result.nit),
                                    "message": str(result.message)})
        return result.x, float(result.fun)

    def fit(self, X, Y, x_scaler=None):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if len(X) != len(Y) or not np.isfinite(Y).all():
            raise ValueError("Invalid training data")
        self.x_scaler = x_scaler if x_scaler is not None else StandardScaler().fit(X)
        self.y_mean = Y.mean(axis=0)
        self.y_scale = safe_scale(Y)
        Z = self.x_scaler.transform(X)
        T = (Y - self.y_mean) / self.y_scale
        self.models, self.fit_warnings, self.optimizer_runs = [], [], []
        for j in range(Y.shape[1]):
            if self.config.kernel == "matern52":
                base = Matern(np.ones(X.shape[1]), self.config.length_bounds, nu=2.5)
            elif self.config.kernel == "rbf":
                base = RBF(np.ones(X.shape[1]), self.config.length_bounds)
            else:
                raise ValueError(self.config.kernel)
            kernel = ConstantKernel(1., self.config.constant_bounds) * base
            kernel += WhiteKernel(self.config.noise_initial, self.config.noise_bounds)
            model = GaussianProcessRegressor(
                kernel=kernel, alpha=self.config.alpha, normalize_y=False,
                optimizer=self._optimizer if self.config.optimize else None,
                n_restarts_optimizer=self.config.restarts, random_state=self.seed + j)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(Z, T[:, j])
            self.fit_warnings.extend({"target_index": j, "message": str(w.message)}
                                     for w in caught)
            self.models.append(model)
        return self

    def predict(self, X):
        Z = self.x_scaler.transform(np.asarray(X, float))
        pairs = [model.predict(Z, return_std=True) for model in self.models]
        mu = np.column_stack([p[0] for p in pairs])
        sigma = np.column_stack([p[1] for p in pairs])
        return mu * self.y_scale + self.y_mean, sigma * self.y_scale

    def diagnostics(self):
        return {"kernels": [str(m.kernel_) for m in self.models],
                "theta": [m.kernel_.theta.tolist() for m in self.models],
                "training_target_mean": self.y_mean.tolist(),
                "training_target_scale": self.y_scale.tolist(),
                "input_mean": self.x_scaler.mean_.tolist(),
                "input_scale": self.x_scaler.scale_.tolist(),
                "warnings": self.fit_warnings, "optimizer_runs": self.optimizer_runs}


def unit_range(values):
    values = np.asarray(values, float)
    extent = np.ptp(values)
    return ((values - values.min()) / extent if extent > 1e-10
            else np.ones_like(values))


def acquisition_scores(strategy, X_pool_scaled, X_train_scaled, normalized_std,
                       iteration, budget, base=0.20, decay=0.60,
                       penalty_mode="weighted", reference_scale=1.0):
    """No candidate or test labels accepted by this interface.

    Multiresponse adaptation averages posterior SD in CURRENT training target SD
    units. One acquisition selects one physical experiment, revealing all outputs.
    DUW weighted branch exactly retains the legacy minmax/offset/weight formula.
    """
    std = np.asarray(normalized_std, float)
    uncertainty = std.mean(axis=1) if std.ndim == 2 else std
    distances = cdist(X_pool_scaled, X_train_scaled).min(axis=1)
    if strategy == "uncertainty":
        return uncertainty
    if strategy == "maximin":
        return distances
    if strategy != "duw":
        raise ValueError(strategy)
    if penalty_mode == "weighted":
        w = base * max(1. - iteration / max(budget, 1) * decay, .05)
        return (unit_range(uncertainty) + .01) ** (1. - w) * (unit_range(distances) + .01) ** w
    radius = base * max(1. - iteration / max(budget, 1) * decay, .1) * reference_scale
    if penalty_mode == "hard":
        return uncertainty * np.where(distances < radius, .001, 1.)
    if penalty_mode == "soft":
        return uncertainty * (1. - np.exp(-distances**2 / (2*radius**2 + 1e-10)))
    raise ValueError(penalty_mode)


def select_next(strategy, X_pool_scaled, X_train_scaled, normalized_std,
                iteration, budget, rng, **kwargs):
    if strategy == "random":
        return int(rng.integers(len(X_pool_scaled))), None
    scores = acquisition_scores(strategy, X_pool_scaled, X_train_scaled,
                                normalized_std, iteration, budget, **kwargs)
    return int(np.argmax(scores)), scores
