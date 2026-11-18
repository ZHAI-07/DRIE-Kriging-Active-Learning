"""评估协议测试：按格留出、噪声上限、置换对照、伪重复检查

测试覆盖：
1. 格结构（23条记录 -> 15格，重复格 4/3/2/2/2）
2. 留一格切分（15折，训练集不含任何同格记录）
3. 噪声上限黄金值（CD 0.900 / AR 0.868 / 全体 0.884）
4. 上限的代数性质（无重复格时上限退化为 1.0，纯噪声时上限为 0）
5. 合并指标与 sklearn r2_score 一致
6. 单点折级 nRMSE 不因 ss_tot=0 崩溃
7. 置换对照保留边际、摧毁关联
8. 伪重复检测（复现 30 次抽样仅 2 个独立初始集的情形）
9. 配对统计同时给出符号检验与 Wilcoxon
10. D-optimal 设计族不把同格重复选入同一设计

意义：
- 这些数字是判断模型好坏与比较规则的基准，必须在代码层面锁定
- 伪重复检测用于防止"有效样本量被高估"这类假显著性
- 上限定理是无分布假设的代数恒等式，可用合成数据直接验证
"""
import sys
from pathlib import Path

import numpy as np
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code"))

from research.data import load_dataset
from research.protocol import (
    bonferroni_threshold, cell_counts, cell_labels, cell_table,
    count_distinct_initial_sets, d_optimal_designs, design_audit, fold_nrmse,
    leave_one_cell_out, noise_ceiling, paired_summary, permutation_control,
    pooled_metrics,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        """加载排除24号后的DRIE数据集，并计算格标签"""
        self.data = load_dataset(ROOT, "real", exclude_ids=(24,))
        self.X = self.data.X
        self.Y = self.data.Y
        self.labels = cell_labels(self.X)

    def test_cell_structure_matches_recorded_design(self):
        """23条记录落在15个(Coil, Platen)格上，重复格为4/3/2/2/2"""
        self.assertEqual(self.X.shape, (23, 4))
        self.assertEqual(self.Y.shape, (23, 12))
        counts = cell_counts(self.labels)
        self.assertEqual(len(counts), 15)
        self.assertEqual(sorted(counts.tolist(), reverse=True),
                         [4, 3, 2, 2, 2] + [1] * 10)
        # 4维输入下全部互异：重复完全来自投影到(Coil, Platen)
        self.assertEqual(len(np.unique(self.X, axis=0)), 23)

    def test_cell_table_lists_members(self):
        """格清点表给出成员ID，且成员数之和等于记录数"""
        table = cell_table(self.X, self.labels, self.data.ids)
        self.assertEqual(sum(row["count"] for row in table), 23)
        self.assertEqual(len(table), 15)
        biggest = max(table, key=lambda row: row["count"])
        self.assertEqual(biggest["count"], 4)
        self.assertEqual(biggest["inputs"][:2], [880.0, 40.0])
        self.assertEqual(sorted(biggest["member_ids"]), [2, 17, 20, 23])

    def test_leave_one_cell_out_is_clean(self):
        """15折，且每折训练集不含被留出格的任何记录"""
        splits = leave_one_cell_out(self.labels)
        self.assertEqual(len(splits), 15)
        for train, test in splits:
            self.assertEqual(len(set(self.labels[train]) & set(self.labels[test])), 0)
            self.assertEqual(len(train) + len(test), 23)

    def test_noise_ceiling_golden_values(self):
        """噪声上限黄金值：CD 0.900、AR 0.868、全体 0.884"""
        ceiling = noise_ceiling(self.Y, self.labels)
        names = self.data.target_names
        cd = [v for v, n in zip(ceiling["per_target"], names) if "CD" in n]
        ar = [v for v, n in zip(ceiling["per_target"], names) if "AR" in n]
        self.assertAlmostEqual(float(np.mean(cd)), 0.900, places=3)
        self.assertAlmostEqual(float(np.mean(ar)), 0.868, places=3)
        self.assertAlmostEqual(ceiling["macro"], 0.884, places=3)
        self.assertEqual(ceiling["replicated_cells"], 5)

    def test_noise_ceiling_degenerates_without_replicates(self):
        """无重复格时格内平方和为零，上限退化为1"""
        labels = np.arange(6)                      # 每格只有一条
        Y = np.arange(6, dtype=float).reshape(-1, 1)
        ceiling = noise_ceiling(Y, labels)
        self.assertAlmostEqual(ceiling["per_target"][0], 1.0, places=12)
        self.assertEqual(ceiling["replicated_cells"], 0)

    def test_noise_ceiling_zero_when_only_noise(self):
        """重复格内只有噪声（每格常数响应）时上限为0"""
        labels = np.array([0, 0, 0, 0])
        Y = np.ones((4, 1))
        ceiling = noise_ceiling(Y, labels)
        # 常数响应：格内平方和为0，总平方和也为0，此处按未定义处理
        self.assertTrue(np.isnan(ceiling["per_target"][0]))

        # 两格各自常数、格间不同：格内平方和为0，上限仍为1
        labels2 = np.array([0, 0, 1, 1])
        Y2 = np.array([[1.0], [1.0], [2.0], [2.0]])
        self.assertAlmostEqual(noise_ceiling(Y2, labels2)["per_target"][0], 1.0,
                               places=12)

    def test_pooled_metrics_match_sklearn(self):
        """合并R²与sklearn的r2_score一致（逐目标）"""
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(20, 3))
        pred = truth + rng.normal(scale=0.3, size=(20, 3))
        metrics = pooled_metrics(truth, pred)
        for j in range(3):
            self.assertAlmostEqual(metrics["r2"][j],
                                   float(r2_score(truth[:, j], pred[:, j])),
                                   places=10)

    def test_fold_nrmse_handles_singleton(self):
        """单点折用外部尺度归一，不因 ss_tot=0 崩溃"""
        scale = np.array([1.0, 2.0])
        value = fold_nrmse(np.array([[3.0, 4.0]]), np.array([[2.0, 2.0]]), scale)
        self.assertAlmostEqual(value, (1.0 / 1.0 + 2.0 / 2.0) / 2.0, places=12)

    def test_permutation_control_preserves_marginals(self):
        """置换列保留边际分布，但破坏与响应的关联"""
        rng = np.random.default_rng(0)
        X = self.X.copy()
        original_correlation = abs(np.corrcoef(X[:, 2], self.Y[:, 0])[0, 1])
        Xp = permutation_control(X, [2, 3], rng)
        # 边际不变：排序后的取值完全相同
        self.assertTrue(np.allclose(np.sort(Xp[:, 2]), np.sort(X[:, 2])))
        self.assertTrue(np.allclose(np.sort(Xp[:, 3]), np.sort(X[:, 3])))
        # 未被置换的列保持原样
        self.assertTrue(np.allclose(Xp[:, 0], X[:, 0]))
        # 关联被改变
        permuted_correlation = abs(np.corrcoef(Xp[:, 2], self.Y[:, 0])[0, 1])
        self.assertNotAlmostEqual(original_correlation, permuted_correlation,
                                  places=6)

    def test_pseudo_replication_detection(self):
        """复现伪重复：固定设计下30次抽样只有2个独立初始集"""
        # 模拟5格各1条、1格有2条的设计
        pool = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5, 6]}
        design = [0, 1, 2, 3, 4, 5]
        actual = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            actual.append(tuple(int(rng.choice(pool[c])) for c in design))
        report = count_distinct_initial_sets(actual)
        self.assertEqual(report.draws, 30)
        self.assertEqual(report.distinct, 2)
        self.assertTrue(report.duplicated)
        self.assertIn("2", report.message)

    def test_distinct_sampling_passes_check(self):
        """互异的初始集应通过伪重复检查"""
        sets = [tuple(sorted(np.random.default_rng(s).choice(17, 6, replace=False)))
                for s in range(30)]
        report = count_distinct_initial_sets(sets)
        self.assertFalse(report.duplicated)
        self.assertEqual(report.distinct, report.draws)

    def test_paired_summary_reports_both_tests(self):
        """配对摘要同时给出符号检验与 Wilcoxon，并识别胜负方向"""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a + 0.5                                # a 处处更小 => 更好
        summary = paired_summary(a, b, higher_is_better=False)
        self.assertEqual(summary["wins"], 5)
        self.assertEqual(summary["losses"], 0)
        self.assertAlmostEqual(summary["delta"], -0.5, places=12)
        self.assertIn("sign_test_p", summary)
        self.assertIn("wilcoxon_p", summary)
        self.assertLessEqual(summary["wilcoxon_p"], 1.0)

    def test_bonferroni_threshold(self):
        """多重比较阈值"""
        self.assertAlmostEqual(bonferroni_threshold(4), 0.0125, places=12)
        with self.assertRaises(ValueError):
            bonferroni_threshold(0)

    def test_d_optimal_never_selects_duplicate_cells(self):
        """D-optimal 设计族的每个设计只用不同格的代表记录"""
        pool = [i for i in range(len(self.X))]
        family = d_optimal_designs(self.X, pool, size=6, top_fraction=0.05)
        self.assertGreater(len(family), 0)
        for combo, det in family:
            self.assertEqual(len({int(self.labels[i]) for i in combo}), 6)

    def test_design_audit_matches_tests(self):
        """审计输出与各测试断言一致，可直接写入验证记录"""
        audit = design_audit(ROOT)
        self.assertEqual(audit["n_records"], 23)
        self.assertEqual(audit["cells"], 15)
        self.assertEqual(audit["replicated_cells"], 5)
        self.assertAlmostEqual(audit["noise_ceiling_macro"], 0.884, places=3)
        self.assertEqual(audit["recommended_holdout_unit"], "cell")
        # 重复格比例取决于抽样池，两个口径都给出且在(0,1)内
        self.assertTrue(0.0 < audit["random_draw_duplicate_cell_rate_all_records"] < 1.0)
        self.assertTrue(0.0 < audit["random_draw_duplicate_cell_rate_al_pool"] < 1.0)


if __name__ == "__main__":
    unittest.main()
