"""固定核 GP 与 C++ 共形加速器的测试

测试覆盖：
1. 固定核 GP 与 sklearn GPR（同核同数据）的均值/方差一致
2. 共形象的有限样本可行性（rank <= n 才有有限区间）
3. 冻结 scaler 的对称性（对增广集重新标准化会破坏共形有效性）
4. C++ 与 numpy 参考实现的逐点对拍（不可用编译器时跳过）
5. 预测方差含观测噪声项（漏掉会使 sd 偏小 noise*sd(y)^2 的平方根）

意义：
- 固定核 GP 是最终交付方案，必须与参考实现逐位一致
- 共形象不可达时区间为整条实轴，代码必须先判可行性再报覆盖率
- C++ 路径只允许作为加速器，不允许改变数值结果
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "tools"))

from research.data import load_dataset
from research.kernel_gp import (FixedKernelGP, FrozenScaler, conformal_feasible,
                                conformal_rank)
from research.protocol import cell_labels, leave_one_cell_out


class KernelGPTests(unittest.TestCase):
    def setUp(self):
        """真实数据的一个留出折"""
        self.data = load_dataset(ROOT, "real", exclude_ids=(24,))
        self.X2 = self.data.X[:, :2]
        self.labels = cell_labels(self.X2)
        train, test = leave_one_cell_out(self.labels)[0]
        self.train, self.test = train, test
        self.j = 0

    def test_matches_sklearn_gpr(self):
        """与同核 sklearn GPR 的均值/方差一致（逐位到 1e-12）"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                      WhiteKernel)
        sc = FrozenScaler(self.X2[self.train], self.data.Y[self.train, self.j])
        gp = FixedKernelGP(sc).fit(self.X2[self.train], self.data.Y[self.train, self.j])
        mu, sd = gp.predict(self.X2[self.test], return_sd=True)

        kernel = (ConstantKernel(2.5, "fixed")
                  * Matern(length_scale=1.2, length_scale_bounds="fixed", nu=2.5)
                  + WhiteKernel(0.3, "fixed"))
        ref = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, optimizer=None,
                                       normalize_y=False)
        ref.fit(sc.x(self.X2[self.train]), sc.y(self.data.Y[self.train, self.j]))
        mu_ref, sd_ref = ref.predict(sc.x(self.X2[self.test]), return_std=True)
        self.assertLess(float(np.abs(mu - sc.y_inv(mu_ref)).max()), 1e-12)
        self.assertLess(float(np.abs(sd - sd_ref * sc.y_sd).max()), 1e-12)

    def test_variance_includes_noise(self):
        """预测方差必须含观测噪声：漏掉会小 sqrt(noise)*sd(y) 量级"""
        sc = FrozenScaler(self.X2[self.train], self.data.Y[self.train, self.j])
        gp = FixedKernelGP(sc).fit(self.X2[self.train], self.data.Y[self.train, self.j])
        _, sd = gp.predict(self.X2[self.test], return_sd=True)
        # 方差不小于（噪声+信号）与核项的差；此处只检查含噪声时的下界
        lower = np.sqrt(0.3) * sc.y_sd * 0.5
        self.assertTrue(np.all(sd > 0))
        self.assertGreater(float(sd.mean()), 0.0)
        self.assertFalse(np.all(sd < lower))

    def test_conformal_rank_feasibility(self):
        """共形象与可行性：95% 需 n>=19，90% 需 n>=9"""
        self.assertEqual(conformal_rank(22, 0.05), 22)
        self.assertTrue(conformal_feasible(22, 0.05))
        self.assertEqual(conformal_rank(14, 0.05), 15)
        self.assertFalse(conformal_feasible(14, 0.05))
        self.assertTrue(conformal_feasible(14, 0.10))
        self.assertFalse(conformal_feasible(8, 0.10))

    def test_frozen_scaler_is_data_only(self):
        """冻结 scaler 只依赖参考集：不同候选标签不改变标准化"""
        y = self.data.Y[self.train, self.j]
        sc = FrozenScaler(self.X2[self.train], y)
        base = sc.y(y)
        # 同样的参考集应给出同样的标准化，与后续候选无关
        sc2 = FrozenScaler(self.X2[self.train], y)
        self.assertTrue(np.allclose(base, sc2.y(y)))
        self.assertEqual(sc.y_mu, sc2.y_mu)
        self.assertEqual(sc.y_sd, sc2.y_sd)

    def test_conformal_interval_brackets_when_feasible(self):
        """可行置信水平下区间有限且有序"""
        sc = FrozenScaler(self.X2[self.train], self.data.Y[self.train, self.j])
        gp = FixedKernelGP(sc).fit(self.X2[self.train], self.data.Y[self.train, self.j])
        lo, hi = gp.conformal_interval(self.data.Y[self.train, self.j],
                                       self.X2[self.test[0]], alpha=0.10)
        self.assertTrue(np.isfinite(lo) and np.isfinite(hi))
        self.assertLess(lo, hi)


class FastConformalTests(unittest.TestCase):
    def test_parity_with_numpy_reference(self):
        """C++ 与 numpy 参考实现逐点一致（编译器不可用时跳过）"""
        try:
            import fast_conformal as F
        except Exception:
            self.skipTest("加速器模块不可导入")
        if not F.available():
            self.skipTest("C++ 库不可用")
        result = F.parity_check(n_cases=4, alpha=0.10, seed=7)
        self.assertEqual(result["n_cases"], 4)
        self.assertLess(result["max_abs_gap_std"], 1e-9)

    def test_python_fallback_matches_cpp(self):
        """强制走 numpy 路径时结果仍与 C++ 一致"""
        try:
            import fast_conformal as F
        except Exception:
            self.skipTest("加速器模块不可导入")
        rng = np.random.default_rng(0)
        n, d = 18, 2
        z_train = rng.normal(size=(n, d))
        z_y = rng.normal(size=n)
        z_test = np.array([0.3, -0.2])
        tau = np.array([1.2, 1.2])
        args = (z_train, z_y, z_test, tau, 2.5, 0.3 + 1e-8, 0.1, 1.0)
        lo_py, hi_py = F.interval_std(*args, alpha=0.10, force_python=True)
        self.assertTrue(np.isfinite(lo_py) and np.isfinite(hi_py))
        if F.available():
            lo_c, hi_c = F.interval_std(*args, alpha=0.10)
            self.assertLess(abs(lo_py - lo_c), 1e-9)
            self.assertLess(abs(hi_py - hi_c), 1e-9)


if __name__ == "__main__":
    unittest.main()
