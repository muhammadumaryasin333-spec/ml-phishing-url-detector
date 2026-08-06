"""Focused tests for the audited Week 6 training protocol."""

import unittest

import numpy as np
import pandas as pd

from src.train_advanced_models import (
    LIGHTGBM_CANDIDATES,
    XGBOOST_CANDIDATES,
    build_advanced_estimator,
    build_advanced_preprocessor,
    calculate_full_metrics,
    positive_class_scores,
    select_best_candidate,
)


class AdvancedTrainingTests(unittest.TestCase):
    """Protect bounded candidates, feature transforms, and metric semantics."""

    def test_candidate_inventory_is_bounded_and_valid(self) -> None:
        self.assertEqual(len(XGBOOST_CANDIDATES), 4)
        self.assertEqual(len(LIGHTGBM_CANDIDATES), 4)
        for parameters in LIGHTGBM_CANDIDATES.values():
            self.assertLessEqual(
                parameters["num_leaves"], 2 ** parameters["max_depth"]
            )
            if parameters["subsample"] < 1:
                self.assertGreater(parameters["subsample_freq"], 0)

    def test_estimators_construct_with_explicit_runtime_controls(self) -> None:
        xgb = build_advanced_estimator("xgboost", XGBOOST_CANDIDATES["xgb_01"])
        lightgbm = build_advanced_estimator(
            "lightgbm", LIGHTGBM_CANDIDATES["lgb_01"]
        )
        self.assertEqual(xgb.get_params()["tree_method"], "hist")
        self.assertEqual(xgb.get_params()["device"], "cpu")
        self.assertTrue(lightgbm.get_params()["deterministic"])
        self.assertTrue(lightgbm.get_params()["force_col_wise"])

    def test_preprocessor_handles_unseen_tld_without_refitting(self) -> None:
        train = pd.DataFrame(
            {"numeric": [1.0, 2.0, 3.0], "TLD": ["com", "org", "com"]}
        )
        validation = pd.DataFrame({"numeric": [4.0], "TLD": ["unseen"]})
        preprocessor = build_advanced_preprocessor(train)
        train_matrix = preprocessor.fit_transform(train)
        validation_matrix = preprocessor.transform(validation)
        self.assertEqual(train_matrix.shape[1], validation_matrix.shape[1])

    def test_selection_uses_metrics_then_candidate_id(self) -> None:
        rows = [
            {
                "candidate_id": "candidate_b",
                "f1_macro": 0.9,
                "phishing_recall": 0.8,
                "roc_auc": 0.95,
            },
            {
                "candidate_id": "candidate_a",
                "f1_macro": 0.9,
                "phishing_recall": 0.8,
                "roc_auc": 0.95,
            },
        ]
        self.assertEqual(select_best_candidate(rows)["candidate_id"], "candidate_a")

    def test_metrics_and_positive_probability_use_phishing_class_zero(self) -> None:
        class FakeModel:
            classes_ = np.array([0, 1])

            @staticmethod
            def predict_proba(X: np.ndarray) -> np.ndarray:
                return np.array([[0.9, 0.1], [0.2, 0.8]])

        scores = positive_class_scores(FakeModel(), np.zeros((2, 1)))
        np.testing.assert_allclose(scores, [0.9, 0.2])
        metrics = calculate_full_metrics(
            pd.Series([0, 1]), np.array([0, 1]), scores
        )
        self.assertEqual(metrics["phishing_recall"], 1.0)
        self.assertEqual(metrics["cm_phishing_as_phishing"], 1)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
