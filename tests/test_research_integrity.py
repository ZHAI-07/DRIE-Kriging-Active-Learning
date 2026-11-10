"""Small meaningful checks: source-blocked splits, hidden-label isolation, DUW parity."""
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
        self.data = load_dataset(ROOT, "real")

    def test_data_shape_and_source_blocks(self):
        self.assertEqual(self.data.X.shape, (24, 4))
        self.assertEqual(self.data.Y.shape, (24, 12))
        groups = source_groups(self.data)
        for seed in range(100):
            split = make_split(self.data, seed, 6, 6, 8)
            a, b, c = split["test"], split["initial"], split["pool"]
            self.assertEqual((len(a), len(b), len(c)), (6, 6, 12))
            self.assertFalse(set(groups[a]) & set(groups[np.r_[b, c]]))
            self.assertEqual(len(set(np.r_[a, b, c])), 24)

    def test_unrevealed_labels_cannot_change_first_query(self):
        split = make_split(self.data, 77, 6, 6, 8)
        a, b, c = split["initial"], split["pool"], split["test"]
        oracle = LabelOracle(self.data, a, b, c)
        train, X, Y = oracle.training()
        gp1 = IndependentGP(GPConfig(optimize=False)).fit(X, Y)
        original_Y = self.data.Y.copy()
        self.data.Y[np.r_[b, c]] = 1e9  # deliberately poison all hidden labels
        _, X2, Y2 = oracle.training()
        gp2 = IndependentGP(GPConfig(optimize=False)).fit(X2, Y2)
        np.testing.assert_allclose(gp1.y_mean, gp2.y_mean)
        np.testing.assert_allclose(gp1.y_scale, gp2.y_scale)
        mu1, sd1 = gp1.predict(self.data.X[b])
        mu2, sd2 = gp2.predict(self.data.X[b])
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(sd1, sd2)
        Zp, Zt = gp1.x_scaler.transform(self.data.X[b]), gp1.x_scaler.transform(X)
        for strategy in ("uncertainty", "maximin", "duw"):
            s1 = select_next(strategy, Zp, Zt, sd1/gp1.y_scale, 0, 8, np.random.default_rng(0))[0]
            s2 = select_next(strategy, Zp, Zt, sd2/gp2.y_scale, 0, 8, np.random.default_rng(0))[0]
            self.assertEqual(s1, s2)
        with self.assertRaises(ValueError):
            oracle.reveal(c[0])
        oracle.reveal(b[0])
        with self.assertRaises(ValueError):
            oracle.reveal(b[0])
        self.data.Y[:] = original_Y

    def test_legacy_weighted_duw_golden_scores(self):
        # Scalar case can be compared directly with the recovered legacy equation.
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
        result = nondominated([[0., 1.], [1., 0.], [2., 2.], [0., 1.]])
        np.testing.assert_array_equal(result, [True, True, False, True])

    def test_all_recovered_legacy_fixtures(self):
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
        data = load_dataset(ROOT, "cfd").subset(np.arange(8))
        with tempfile.TemporaryDirectory(dir=ROOT/"_tmp", prefix="scalar_test_") as directory:
            metrics = basic_evaluation(data, directory, GPConfig(optimize=False))
            self.assertEqual(set(metrics.model), {"Mean", "Ridge", "GPR"})
            self.assertTrue(metrics.n_predictions.eq(8).all())
            self.assertTrue(np.isfinite(metrics.rmse).all())


if __name__ == "__main__":
    unittest.main()
