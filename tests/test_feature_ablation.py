"""Focused tests for the URLSimilarityIndex feature-ablation setup."""

import unittest

import pandas as pd

from src.run_feature_ablation import create_feature_sets


class FeatureAblationTests(unittest.TestCase):
    """Ensure matched ablation conditions differ only by the audited feature."""

    def test_removes_only_url_similarity_index(self) -> None:
        features = {
            "train": pd.DataFrame(
                {
                    "URLSimilarityIndex": [1.0, 2.0],
                    "URLLength": [10, 20],
                }
            ),
            "validation": pd.DataFrame(
                {
                    "URLSimilarityIndex": [3.0],
                    "URLLength": [30],
                }
            ),
            "test": pd.DataFrame(
                {
                    "URLSimilarityIndex": [4.0],
                    "URLLength": [40],
                }
            ),
        }

        feature_sets = create_feature_sets(features)

        self.assertEqual(
            feature_sets["with_url_similarity_index"]["train"].columns.tolist(),
            ["URLSimilarityIndex", "URLLength"],
        )
        self.assertEqual(
            feature_sets["without_url_similarity_index"]["train"].columns.tolist(),
            ["URLLength"],
        )
        self.assertEqual(
            feature_sets["without_url_similarity_index"]["test"].columns.tolist(),
            ["URLLength"],
        )


if __name__ == "__main__":
    unittest.main()
