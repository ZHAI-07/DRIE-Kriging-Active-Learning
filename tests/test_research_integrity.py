"""研究完整性测试：来源阻断、标签隔离、DUW一致性

测试覆盖：
1. 数据形状与来源分组（24×4输入，24×12输出，2/13同组）
2. 未揭示标签隔离（毒化候选池不影响第一次查询）
3. 遗留DUW加权公式的精确复现（黄金值检验）
4. Pareto前沿计算（不移除相等行）
5. 标量响应的基础评估（CFD数据集）
6. 历史遗留夹具的完整复现（5个案例）

意义：
- 防止标签泄漏（主动学习的核心假设）
- 验证DUW公式与遗留实现完全一致
- 确保数学正确性（来源阻断、Pareto支配）
"""
import sys
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))
from research.core import GPConfig, IndependentGP, acquisition_scores, select_next
from research.data import LabelOracle, load_dataset, make_split, source_groups
from research.optimization import nondominated
from research.experiments import basic_evaluation


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        """加载DRIE真实数据集"""
        self.data = load_dataset(ROOT, "real")

    def test_data_shape_and_source_blocks(self):
        """测试数据形状和来源阻断分割（100个随机种子）"""
        self.assertEqual(self.data.X.shape, (24, 4))
        self.assertEqual(self.data.Y.shape, (24, 12))
        groups = source_groups(self.data)
        for seed in range(100):
            split = make_split(self.data, seed, 6, 6, 8)
            a, b, c = split["test"], split["initial"], split["pool"]
            self.assertEqual((len(a), len(b), len(c)), (6, 6, 12))
            # 来源组无交集：测试集与训练侧（初始+候选）不共享组
            self.assertFalse(set(groups[a]) & set(groups[np.r_[b, c]]))
            self.assertEqual(len(set(np.r_[a, b, c])), 24)

    def test_unrevealed_labels_cannot_change_first_query(self):
        """测试标签隔离：毒化候选池和测试集不影响第一次查询
        
        验证：
        - 候选池/测试集标签不参与GP训练
        - 所有策略的第一次选择完全相同
        - Oracle拒绝揭示测试集和已揭示的点
        """
        split = make_split(self.data, 77, 6, 6, 8)
        a, b, c = split["initial"], split["pool"], split["test"]
        oracle = LabelOracle(self.data, a, b, c)
        train, X, Y = oracle.training()
        gp1 = IndependentGP(GPConfig(optimize=False)).fit(X, Y)
        original_Y = self.data.Y.copy()
        self.data.Y[np.r_[b, c]] = 1e9  # 故意毒化所有隐藏标签
        _, X2, Y2 = oracle.training()
        gp2 = IndependentGP(GPConfig(optimize=False)).fit(X2, Y2)
        # 归一化参数应完全相同（仅依赖已揭示标签）
        np.testing.assert_allclose(gp1.y_mean, gp2.y_mean)
        np.testing.assert_allclose(gp1.y_scale, gp2.y_scale)
        # 预测结果完全相同
        mu1, sd1 = gp1.predict(self.data.X[b])
        mu2, sd2 = gp2.predict(self.data.X[b])
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(sd1, sd2)
        # 所有策略的采样选择完全相同
        Zp, Zt = gp1.x_scaler.transform(self.data.X[b]), gp1.x_scaler.transform(X)
        for strategy in ("uncertainty", "maximin", "duw"):
            s1 = select_next(strategy, Zp, Zt, sd1/gp1.y_scale, 0, 8, np.random.default_rng(0))[0]
            s2 = select_next(strategy, Zp, Zt, sd2/gp2.y_scale, 0, 8, np.random.default_rng(0))[0]
            self.assertEqual(s1, s2)
        # Oracle拒绝非法操作
        with self.assertRaises(ValueError):
            oracle.reveal(c[0])  # 拒绝揭示测试集
        oracle.reveal(b[0])
        with self.assertRaises(ValueError):
            oracle.reveal(b[0])  # 拒绝重复揭示
        self.data.Y[:] = original_Y

    def test_legacy_weighted_duw_golden_scores(self):
        """测试遗留加权DUW公式的精确复现（黄金值）
        
        标量情况可直接与恢复的遗留方程比较
        w = base * (1 - decay * iteration/budget)
        score = (u_norm + 0.01)^(1-w) * (d_norm + 0.01)^w
        """
        # 标量case：3个候选点，1个训练点
        points = np.array([[.2], [.5], [.9]])
        train = np.array([[0.]])
        std = np.array([.1, .4, .2])
        scores = acquisition_scores("duw", points, train, std, iteration=2, budget=10)
        w = .2 * (1 - .6 * 2/10)
        expected = np.array([.01**(1-w)*.01**w,
                             1.01**(1-w)*(3/7+.01)**w,
                             (1/3+.01)**(1-w)*1.01**w])
        np.testing.assert_allclose(scores, expected, rtol=1e-12)

    def test_pareto_does_not_remove_equal_rows(self):
        """测试Pareto前沿不移除相等行（重复解仍保留）"""
        result = nondominated([[0., 1.], [1., 0.], [2., 2.], [0., 1.]])
        np.testing.assert_array_equal(result, [True, True, False, True])

    def test_all_recovered_legacy_fixtures(self):
        """测试所有恢复的遗留夹具（5个历史案例完整复现）"""
        fixtures = json.loads((ROOT/"outputs/reports/duw_recovery.json").read_text(encoding="utf-8"))["acquisition_formula_fixtures"]
        self.assertEqual(len(fixtures), 5)
        for case in fixtures:
            with self.subTest(case=case["name"]):
                scores = acquisition_scores("duw", np.array(case["candidate_x"]),
                                            np.array(case["train_x"]),
                                            np.array(case["predictive_std"]),
                                            case["iteration"], case["budget"])
                np.testing.assert_allclose(scores, case["scores"], rtol=1e-12)
                self.assertEqual(int(np.argmax(scores)), case["legacy_function_selected_index"])

    def test_scalar_response_basic_evaluation(self):
        """测试标量响应的基础评估（CFD数据集8样本子集）"""
        data = load_dataset(ROOT, "cfd").subset(np.arange(8))
        with tempfile.TemporaryDirectory(dir=ROOT/"_tmp", prefix="scalar_test_") as directory:
            metrics = basic_evaluation(data, directory, GPConfig(optimize=False))
            self.assertEqual(set(metrics.model), {"Mean", "Ridge", "GPR"})
            self.assertTrue(metrics.n_predictions.eq(8).all())
            self.assertTrue(np.isfinite(metrics.rmse).all())


if __name__ == "__main__":
    unittest.main()
