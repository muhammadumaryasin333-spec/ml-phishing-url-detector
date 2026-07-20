"""Focused tests for Week 5 baseline training safeguards."""

import unittest

import pandas as pd

from src.train_baseline_models import (
    build_models,
    calculate_metrics,
    exclude_train_only_placeholder_columns,
    find_target_column,
    get_roc_positive_label,
    validate_and_split_datasets,
)


class BaselineTrainingTests(unittest.TestCase):
    """Validate target detection and safe mixed-feature pipeline behaviour."""

    def test_finds_supported_target_columns_case_insensitively(self) -> None:
        self.assertEqual(find_target_column(pd.DataFrame({"Class": [0, 1]})), "Class")
        self.assertEqual(find_target_column(pd.DataFrame({"TARGET": [0, 1]})), "TARGET")
        self.assertIsNone(find_target_column(pd.DataFrame({"outcome": [0, 1]})))

    def test_pipeline_handles_unseen_categorical_values(self) -> None:
        datasets = {
            "train": pd.DataFrame(
                {
                    "numeric_feature": [1.0, 2.0, 3.0, 4.0],
                    "categorical_feature": ["a", "b", "a", None],
                    "label": [0, 1, 0, 1],
                }
            ),
            "validation": pd.DataFrame(
                {
                    "numeric_feature": [2.0, None],
                    "categorical_feature": ["b", "unseen"],
                    "label": [1, 0],
                }
            ),
            "test": pd.DataFrame(
                {
                    "numeric_feature": [3.0, 4.0],
                    "categorical_feature": ["a", "unseen"],
                    "label": [0, 1],
                }
            ),
        }
        validated = validate_and_split_datasets(datasets)
        self.assertIsNotNone(validated)
        features, targets, _ = validated

        model = build_models(features["train"], class_weight="balanced")[
            "logistic_regression"
        ]
        decision_tree = build_models(features["train"], class_weight="balanced")[
            "decision_tree"
        ]
        self.assertIn(
            "scaler",
            model.named_steps["preprocessor"].transformers[0][1].named_steps,
        )
        self.assertNotIn(
            "scaler",
            decision_tree.named_steps["preprocessor"]
            .transformers[0][1]
            .named_steps,
        )
        model.fit(features["train"], targets["train"])
        predicted = model.predict(features["test"])
        positive_label = get_roc_positive_label(targets["train"])
        scores = model.predict_proba(features["test"])[
            :, list(model.classes_).index(positive_label)
        ]
        metrics = calculate_metrics(targets["test"], predicted, scores, positive_label)

        self.assertEqual(
            set(metrics), {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
        )

    def test_excludes_train_only_high_cardinality_placeholder_encodings(self) -> None:
        features = {
            "train": pd.DataFrame(
                {
                    "encoded_url": range(100),
                    "valid_measurement": [index % 2 for index in range(100)],
                    "negative_one_is_valid": [
                        -1 if index == 0 else index for index in range(100)
                    ],
                }
            ),
            "validation": pd.DataFrame(
                {
                    "encoded_url": [-1] * 8 + [1, 2],
                    "valid_measurement": [0, 1] * 5,
                    "negative_one_is_valid": [-1] + list(range(1, 10)),
                }
            ),
            "test": pd.DataFrame(
                {
                    "encoded_url": [-1] * 9 + [3],
                    "valid_measurement": [0, 1] * 5,
                    "negative_one_is_valid": [-1] + list(range(1, 10)),
                }
            ),
        }

        usable_features, excluded_columns = exclude_train_only_placeholder_columns(features)

        self.assertEqual(set(excluded_columns), {"encoded_url"})
        self.assertNotIn("encoded_url", usable_features["train"])
        self.assertIn("valid_measurement", usable_features["train"])
        self.assertIn("negative_one_is_valid", usable_features["train"])


if __name__ == "__main__":
    unittest.main()
