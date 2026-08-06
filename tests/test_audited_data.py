"""Protocol tests for domain-group-disjoint audited data."""

import unittest

import pandas as pd

from src.audited_data import (
    IDENTIFIER_COLUMNS,
    QUARANTINED_FEATURES,
    build_audited_splits,
    normalize_hostname,
    normalize_groups,
    prepare_model_data,
    validate_audited_splits,
)


class AuditedDataTests(unittest.TestCase):
    """Protect grouping, holdout isolation, and feature quarantine rules."""

    @staticmethod
    def build_dataset() -> pd.DataFrame:
        rows = []
        for domain_index in range(40):
            for row_index in range(4):
                label = (domain_index + row_index) % 2
                rows.append(
                    {
                        "FILENAME": f"row-{domain_index}-{row_index}",
                        "URL": f"https://site{domain_index}.example/{row_index}",
                        "Domain": f" Site{domain_index}.Example ",
                        "Title": f"Title {domain_index}",
                        "TLD": "example",
                        "URLSimilarityIndex": 100 if label else 20,
                        "CharContinuationRate": 0.5,
                        "TLDLegitimateProb": 0.8,
                        "URLCharProb": 0.7,
                        "URLLength": 20 + row_index,
                        "label": label,
                    }
                )
        return pd.DataFrame(rows)

    def test_normalizes_domain_groups(self) -> None:
        normalized = normalize_groups(
            pd.Series([" Example.COM ", "www.EXAMPLE.com.", "user@example.com:443"])
        )
        self.assertEqual(
            normalized.tolist(), ["example.com", "example.com", "example.com"]
        )
        self.assertEqual(normalize_hostname("127.000.000.001"), "127.000.000.001")

    def test_group_splits_have_zero_domain_and_url_overlap(self) -> None:
        splits = build_audited_splits(self.build_dataset())
        validate_audited_splits(splits)

        domain_sets = {
            name: set(normalize_groups(split["Domain"]))
            for name, split in splits.items()
        }
        self.assertFalse(domain_sets["train"] & domain_sets["validation"])
        self.assertFalse(domain_sets["train"] & domain_sets["test"])
        self.assertFalse(domain_sets["validation"] & domain_sets["test"])

    def test_input_row_order_does_not_change_membership(self) -> None:
        dataset = self.build_dataset()
        original = build_audited_splits(dataset)
        shuffled = build_audited_splits(dataset.sample(frac=1, random_state=7))

        for split_name in ["train", "validation", "test"]:
            self.assertEqual(
                set(original[split_name]["FILENAME"]),
                set(shuffled[split_name]["FILENAME"]),
            )

    def test_model_data_excludes_identifiers_and_quarantined_features(self) -> None:
        splits = build_audited_splits(self.build_dataset())
        features, targets = prepare_model_data(splits)

        excluded = IDENTIFIER_COLUMNS | QUARANTINED_FEATURES
        self.assertFalse(excluded & set(features["train"].columns))
        self.assertIn("TLD", features["train"].columns)
        self.assertIn("URLLength", features["train"].columns)
        self.assertEqual(set(targets), {"train", "validation", "test"})

    def test_rejects_cross_split_domain_overlap(self) -> None:
        splits = build_audited_splits(self.build_dataset())
        splits["test"].loc[0, "Domain"] = splits["train"].loc[0, "Domain"]

        with self.assertRaisesRegex(ValueError, "Domain overlap"):
            validate_audited_splits(splits)

    def test_rejects_domain_and_url_hostname_mismatch(self) -> None:
        dataset = self.build_dataset()
        dataset.loc[0, "Domain"] = "different.example"

        with self.assertRaisesRegex(ValueError, "hostnames disagree"):
            build_audited_splits(dataset)


if __name__ == "__main__":
    unittest.main()
