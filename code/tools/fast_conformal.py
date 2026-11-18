"""全共形反演：C++ 加速器 + numpy 参考实现

设计取舍
--------
只有全共形的候选反演用 C++ 重写。单次 GP 拟合（n<=24）在 numpy 里约 16.6 µs，
大半是 scipy 调用开销，移植收益远小于维护成本；而一次共形区间要做约 420 次
小规模 Cholesky（实测 6.9 ms），覆盖率自助区间会调用几十万次，这才是热点。

C++ 不可用（无编译器、未构建）时自动回退到本文件内的 numpy 参考实现：慢，但
结果一致。两条路径的对拍不是断言而是实测（`parity_check`），差异超过 1e-9 即
视为不一致。

数学约定
--------
    核：k(x,x') = signal * f(s),  s = sqrt(5) * ||(x-x')/tau_dim||
        f(s) = (1 + s + s^2/3) exp(-s)                     （Matern 5/2）
    训练协方差：K = k(Z,Z) + (noise + jitter) I            （对角线含观测噪声）
    预测行：无噪声的 k(Z_test, Z)                          （不能用 K，否则残差恒 0）
    共形象：rank = ceil((n+1)(1-alpha))，与其它 n 个分数比较
    接受：|z_cand - fit_cand| <= 第 rank 小的 |z_i - fit_i|

所有输入输出都在已标准化坐标（X 与 y 由调用方用冻结 scaler 标准化）；本模块
不做标准化，避免与调用方口径漂移。
"""
from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "conformal_kernel.cpp"
LIBRARY = HERE / ("conformal_kernel.dylib" if platform.system() == "Darwin"
                  else "conformal_kernel.so")


# ------------------------------------------------------------- 构建与加载
def build(force=False):
    """编译共享库；失败抛 RuntimeError 并保留编译器输出"""
    if LIBRARY.exists() and not force:
        return LIBRARY
    compiler = "clang++" if platform.system() == "Darwin" else "g++"
    cmd = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
           str(SOURCE), "-o", str(LIBRARY)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"编译失败（{compiler}）：\n{result.stderr}")
    return LIBRARY


def _load():
    if not LIBRARY.exists():
        try:
            build()
        except Exception:
            return None
    try:
        lib = ctypes.CDLL(str(LIBRARY))
    except OSError:
        return None
    dbl_p = ctypes.POINTER(ctypes.c_double)
    lib.full_conformal_interval_std.argtypes = [
        ctypes.c_int, ctypes.c_int, dbl_p, dbl_p, dbl_p, dbl_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        dbl_p, dbl_p,
    ]
    lib.full_conformal_interval_std.restype = ctypes.c_int
    return lib


_LIB = _load()


def available():
    """C++ 路径是否可用"""
    return _LIB is not None


# --------------------------------------------------------- numpy 参考实现
def _matern52(s):
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


def _gram(A, B, tau, signal):
    diff = (A[:, None, :] - B[None, :, :]) / np.asarray(tau)[None, None, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return signal * _matern52(np.sqrt(5.0) * d)


def _margin(z_train, z_y, z_test, tau, signal, noise, z_cand, rank_k):
    """候选标签的 (自身残差 - 第 rank 小训练残差)；None 表示非正定"""
    n = len(z_y)
    Z_aug = np.vstack([z_train, z_test[None, :]])
    z_aug = np.append(z_y, z_cand)
    K = _gram(Z_aug, Z_aug, tau, signal)
    K = K + noise * np.eye(n + 1)
    try:
        chol = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return None
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, z_aug))
    # 预测用无噪声核行：K 的对角含噪声，统一扣 noise*alpha_i
    fitted = K @ alpha - noise * alpha
    residual = np.abs(z_aug - fitted)
    if rank_k > n:
        return 1e300
    threshold = np.partition(residual[:n], rank_k - 1)[rank_k - 1]
    return float(residual[n] - threshold)


def numpy_interval_std(z_train, y_train_std, z_test, tau, signal, noise,
                       center, spread, alpha=0.10, grid=65, span=6.0,
                       max_expand=6):
    """numpy 参考实现：候选网格扫描 + 双侧二分，与 C++ 版本同一构造"""
    n = int(len(y_train_std))
    rank_k = int(np.ceil((n + 1) * (1.0 - alpha)))
    scale = spread if spread > 0 else 1.0
    cand = accepted = None
    first = last = -1
    for _ in range(max_expand):
        cand = center + np.linspace(-span, span, grid) * scale
        margins = np.array([
            _margin(z_train, y_train_std, z_test, tau, signal, noise, c, rank_k)
            for c in cand])
        if np.any(margins == None):  # noqa: E711  - 非正定
            return float("nan"), float("nan")
        accepted = margins <= 0.0
        if accepted.any() and not accepted[0] and not accepted[-1]:
            first = int(np.flatnonzero(accepted)[0])
            last = int(np.flatnonzero(accepted)[-1])
            break
        scale *= 2.0
    if first < 0:
        return float("nan"), float("nan")

    def bisect(inside, outside):
        ok = _margin(z_train, y_train_std, z_test, tau, signal, noise,
                     inside, rank_k) <= 0.0
        for _ in range(50):
            mid = 0.5 * (inside + outside)
            mg = _margin(z_train, y_train_std, z_test, tau, signal, noise,
                         mid, rank_k)
            if (mg <= 0.0) == ok:
                inside = mid
            else:
                outside = mid
        return inside if ok else outside

    left = bisect(cand[first], cand[first - 1]) if first > 0 else cand[first]
    right = bisect(cand[last], cand[last + 1]) if last < len(cand) - 1 else cand[last]
    return float(left), float(right)


# --------------------------------------------------------------- 统一入口
def interval_std(z_train, y_train_std, z_test, tau, signal, noise,
                 center, spread, alpha=0.10, grid=65, span=6.0, max_expand=6,
                 force_python=False):
    """求一个全共形区间（标准化坐标）；C++ 可用即用之，否则回退 numpy"""
    if _LIB is None or force_python:
        return numpy_interval_std(z_train, y_train_std, z_test, tau, signal,
                                  noise, center, spread, alpha, grid, span,
                                  max_expand)
    z_train = np.ascontiguousarray(z_train, dtype=np.float64)
    y_train_std = np.ascontiguousarray(y_train_std, dtype=np.float64)
    z_test = np.ascontiguousarray(z_test, dtype=np.float64)
    tau = np.ascontiguousarray(tau, dtype=np.float64)
    n, d = z_train.shape
    left = ctypes.c_double(0.0)
    right = ctypes.c_double(0.0)
    status = _LIB.full_conformal_interval_std(
        int(n), int(d),
        z_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_train_std.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(signal), float(noise), float(center), float(spread),
        int(grid), float(span), float(alpha), int(max_expand),
        ctypes.byref(left), ctypes.byref(right))
    if status != 1:
        return float("nan"), float("nan")
    return left.value, right.value


# ------------------------------------------------------------------- 对拍
def parity_check(n_cases=12, alpha=0.10, seed=0, verbose=False):
    """C++ 与 numpy 参考实现的逐点对拍 + 速度对比

    返回 max_abs_gap（标准化单位下的最大端点差）、两路耗时与加速比。
    """
    import sys
    import time

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "code"))
    from research.data import load_dataset
    from research.protocol import cell_labels

    data = load_dataset(root, "real", exclude_ids=(24,))
    X2 = data.X[:, :2]
    labels = cell_labels(X2)
    rng = np.random.default_rng(seed)

    gaps, t_cpp, t_py = [], 0.0, 0.0
    for _ in range(n_cases):
        cell = int(rng.choice(np.unique(labels)))
        train = np.flatnonzero(labels != cell)
        test = np.flatnonzero(labels == cell)
        j = int(rng.integers(0, data.Y.shape[1]))
        x_mu, x_sd = X2[train].mean(axis=0), X2[train].std(axis=0, ddof=0)
        x_sd = np.where(x_sd > 1e-12, x_sd, 1.0)
        y_mu = float(data.Y[train, j].mean())
        y_sd = float(data.Y[train, j].std(ddof=0)) or 1.0
        Z = (X2[train] - x_mu) / x_sd
        Zt = (X2[test[0]] - x_mu) / x_sd
        z_y = (data.Y[train, j] - y_mu) / y_sd

        # 候选网格中心与尺度：用同一个训练子集上的点预测与后验 sd
        tau = np.array([1.2, 1.2])
        signal, noise = 2.5, 0.3 + 1e-8
        K = _gram(Z, Z, tau, signal) + noise * np.eye(len(Z))
        alpha_v = np.linalg.solve(K, z_y)
        ks = _gram(Zt[None, :], Z, tau, signal)[0]
        mu_z = float(ks @ alpha_v)
        var = signal + noise - float(ks @ np.linalg.solve(K, ks))
        sd_z = float(np.sqrt(max(var, 1e-12)))
        center = mu_z
        spread = max(sd_z, 1e-6)

        t0 = time.perf_counter()
        c_left, c_right = interval_std(Z, z_y, Zt, tau, signal, noise,
                                       center, spread, alpha=alpha)
        t_cpp += time.perf_counter() - t0

        t0 = time.perf_counter()
        p_left, p_right = numpy_interval_std(Z, z_y, Zt, tau, signal, noise,
                                             center, spread, alpha=alpha)
        t_py += time.perf_counter() - t0

        if np.isnan(c_left) or np.isnan(p_left):
            continue
        gaps.append(max(abs(c_left - p_left), abs(c_right - p_right)))
        if verbose:
            print(f"  case {_:2d}  cpp [{c_left:+.6f},{c_right:+.6f}]  "
                  f"py [{p_left:+.6f},{p_right:+.6f}]  gap {gaps[-1]:.2e}")

    return {"n_cases": len(gaps),
            "max_abs_gap_std": float(np.max(gaps)) if gaps else float("nan"),
            "seconds_cpp": t_cpp, "seconds_python": t_py,
            "speedup": t_py / max(t_cpp, 1e-12),
            "c_available": available()}


if __name__ == "__main__":
    print(f"库路径：{LIBRARY}")
    print(f"C++ 可用：{available()}")
    print(parity_check(verbose=False))
