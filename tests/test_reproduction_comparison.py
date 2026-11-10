"""Portable reproduction accepts numerical drift but never changed queries."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_submission as submission


class ReproductionComparisonTests(unittest.TestCase):
    def setUp(self):
        (ROOT/"_tmp").mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/"_tmp",prefix="comparison-test-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patch=patch.object(submission,"ROOT",self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.source=self.root/"outputs/results/reference";self.target=self.root/"outputs/results/repeated"
        self.source.mkdir(parents=True);self.target.mkdir(parents=True)
        self.curves=pd.DataFrame({"seed":[1,1],"strategy":["duw","duw"],"target":["AR","AR"],"step":[0,1],
                                  "n_train":[6,7],"structure":["3-3","3-3"],"metric":["AR","AR"],
                                  "rmse":[.6,.59],"nrmse":[1.,.98]})
        self.auc=pd.DataFrame({"seed":[1],"strategy":["duw"],"target":["AR"],"structure":["3-3"],"metric":["AR"],
                               "test_auc":[.595],"normalized_test_auc":[.99]})
        self.selected=pd.DataFrame({"seed":[1],"strategy":["duw"],"acquisition_step":[1],"experiment_id":[24],
                                   "n_train_before":[6],"remaining_candidates_before":[12],"score":[.4]})
        for name,frame in [("al_curves.csv",self.curves),("al_auc.csv",self.auc),("al_selections.csv",self.selected)]:
            frame.to_csv(self.source/name,index=False);frame.to_csv(self.target/name,index=False)

    def mutate(self,name,callback):
        path=self.target/name;frame=pd.read_csv(path);callback(frame);frame.to_csv(path,index=False)

    def compare(self,**kw):return submission.compare_runs("reference","repeated",**kw)

    def test_platform_sized_curve_difference_and_small_auc_error_pass(self):
        self.mutate("al_curves.csv",lambda x:x.__setitem__("rmse",x.rmse+.015))
        self.mutate("al_auc.csv",lambda x:x.__setitem__("test_auc",x.test_auc*1.02))
        records=self.compare()
        self.assertTrue(all(r["status"]=="passed" for r in records))
        self.assertAlmostEqual(records[0]["max_numeric_difference"],.015)
        self.assertAlmostEqual(records[1]["auc_max_relative_difference"],.02)
        self.assertEqual(records[2]["query_mismatches"],0)
        self.assertEqual(records[1]["relative_denominator_epsilon"],1e-12)

    def test_strict_numeric_mode_rejects_curve_drift(self):
        self.mutate("al_curves.csv",lambda x:x.__setitem__("rmse",x.rmse+.015))
        with self.assertRaises(submission.ReproductionComparisonError) as failure:self.compare(mode="strict")
        self.assertEqual(failure.exception.checks[0]["status"],"failed")

    def test_selected_id_difference_is_rejected(self):
        self.mutate("al_selections.csv",lambda x:x.__setitem__("experiment_id",[23]))
        with self.assertRaises(submission.ReproductionComparisonError) as failure:self.compare()
        self.assertEqual(failure.exception.checks[-1]["query_mismatches"],1)

    def test_auc_above_relative_tolerance_is_rejected(self):
        self.mutate("al_auc.csv",lambda x:x.__setitem__("test_auc",x.test_auc*1.0301))
        with self.assertRaises(submission.ReproductionComparisonError) as failure:self.compare()
        self.assertGreater(failure.exception.checks[1]["auc_max_relative_difference"],.03)

    def test_exact_relative_boundary_passes_despite_roundoff(self):
        self.mutate("al_auc.csv",lambda x:x.__setitem__("test_auc",x.test_auc*1.03))
        self.assertEqual(self.compare()[1]["status"],"passed")

    def test_nonfinite_and_missing_row_are_rejected(self):
        for edit in [lambda x:x.__setitem__("rmse",[np.nan,.59]),lambda x:x.drop(index=0,inplace=True)]:
            with self.subTest(edit=edit):
                self.curves.to_csv(self.target/"al_curves.csv",index=False)
                self.mutate("al_curves.csv",edit)
                with self.assertRaises(submission.ReproductionComparisonError):self.compare()

    def test_duplicate_or_changed_structure_is_rejected(self):
        for frame in [pd.concat([self.curves,self.curves.iloc[[0]]],ignore_index=True),self.curves.assign(structure=["5-5","3-3"])]:
            with self.subTest(rows=len(frame)):
                frame.to_csv(self.target/"al_curves.csv",index=False)
                with self.assertRaises(submission.ReproductionComparisonError):self.compare()

    def test_zero_reference_uses_explicit_epsilon_and_rejects_drift(self):
        for folder in [self.source,self.target]:
            self.auc.assign(test_auc=[0.]).to_csv(folder/"al_auc.csv",index=False)
        self.mutate("al_auc.csv",lambda x:x.__setitem__("test_auc",[1e-8]))
        with self.assertRaises(submission.ReproductionComparisonError) as failure:self.compare()
        self.assertAlmostEqual(failure.exception.checks[1]["auc_max_relative_difference"],1e4)

    def test_first_comparison_failure_does_not_skip_second_scenario(self):
        failed=submission.ReproductionComparisonError([{"file":"al_auc.csv","status":"failed","error":"too large"}])
        out=self.root/"validation"
        with patch.object(submission,"OUT",out),patch.object(sys,"argv",["run_submission.py","--recompute-original","--run-prefix","trial"]), \
             patch.object(submission.platform,"platform",return_value="test-platform"), \
             patch.object(submission.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as run, \
             patch.object(submission,"compare_runs",side_effect=[failed,[{"status":"passed"}]]), \
             patch.object(submission,"paired_macro_diagnostics",return_value=[]):
            with self.assertRaisesRegex(RuntimeError,"both data scenarios"):submission.main()
        invoked=[call.args[0] for call in run.call_args_list]
        self.assertEqual(len(invoked),2)
        self.assertIn("trial_real_main",invoked[0]);self.assertIn("trial_real_drop24",invoked[1])
        record=json.loads((out/"submission_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"],"failed")
        self.assertEqual([x["status"] for x in record["original_protocol_reproduction"]],["failed","passed"])


if __name__=="__main__":unittest.main()
