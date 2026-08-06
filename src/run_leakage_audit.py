"""Create leakage-audited splits and evidence before Week 6 training."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __package__:
    from .audited_data import (
        AUDIT_REPORT_DIR,
        AUDITED_DATA_DIR,
        AUDITED_SPLIT_PATHS,
        GROUP_COLUMN,
        IDENTIFIER_COLUMNS,
        PROJECT_ROOT,
        QUARANTINED_FEATURES,
        RANDOM_STATE,
        RAW_DATA_PATH,
        TARGET_COLUMN,
        build_audited_splits,
        normalize_groups,
        write_split_manifest,
    )
else:
    from audited_data import (
        AUDIT_REPORT_DIR,
        AUDITED_DATA_DIR,
        AUDITED_SPLIT_PATHS,
        GROUP_COLUMN,
        IDENTIFIER_COLUMNS,
        PROJECT_ROOT,
        QUARANTINED_FEATURES,
        RANDOM_STATE,
        RAW_DATA_PATH,
        TARGET_COLUMN,
        build_audited_splits,
        normalize_groups,
        write_split_manifest,
    )


SPLIT_SUMMARY_PATH = AUDIT_REPORT_DIR / "split_summary.csv"
LEGACY_OVERLAP_PATH = AUDIT_REPORT_DIR / "legacy_split_overlap.csv"
UNIVARIATE_RESULTS_PATH = AUDIT_REPORT_DIR / "univariate_validation_results.csv"
PROVENANCE_PATH = AUDIT_REPORT_DIR / "feature_provenance.csv"
REPORT_PATH = AUDIT_REPORT_DIR / "leakage_audit_report.md"
UNIVARIATE_AUC_FLAG_THRESHOLD = 0.98


def build_split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize row, group, and class distributions for audited splits."""
    rows = []
    for split_name, split in splits.items():
        counts = split[TARGET_COLUMN].value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "rows": len(split),
                    "domains": normalize_groups(split[GROUP_COLUMN]).nunique(),
                    "class": int(label),
                    "class_count": int(count),
                    "class_percentage": count / len(split) * 100,
                }
            )
    return pd.DataFrame(rows)


def reconstruct_legacy_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recreate the legacy row-random split for overlap measurement only."""
    train, temporary = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary[TARGET_COLUMN],
    )
    return {"train": train, "validation": validation, "test": test}


def build_overlap_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Measure URL and exact-domain membership overlap between split pairs."""
    rows = []
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for entity_column in ["URL", GROUP_COLUMN]:
        entity_sets = {
            split_name: set(
                normalize_groups(split[entity_column])
                if entity_column == GROUP_COLUMN
                else split[entity_column].astype("string")
            )
            for split_name, split in splits.items()
        }
        for left, right in pairs:
            shared = entity_sets[left] & entity_sets[right]
            right_values = (
                normalize_groups(splits[right][entity_column])
                if entity_column == GROUP_COLUMN
                else splits[right][entity_column].astype("string")
            )
            right_mask = right_values.isin(shared)
            rows.append(
                {
                    "entity": entity_column,
                    "left_split": left,
                    "right_split": right,
                    "shared_unique_values": len(shared),
                    "right_rows_with_overlap": int(right_mask.sum()),
                    "right_overlap_percentage": right_mask.mean() * 100,
                }
            )
    return pd.DataFrame(rows)


def audit_univariate_features(
    train: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    """Rank numeric features using train-fitted one-feature validation models."""
    result_rows = []
    excluded = IDENTIFIER_COLUMNS | {TARGET_COLUMN}
    numeric_columns = train.drop(columns=sorted(excluded)).select_dtypes(
        include="number"
    ).columns
    for column in numeric_columns:
        train_values = train[[column]].copy()
        validation_values = validation[[column]].copy()
        median = train_values[column].median()
        train_values[column] = train_values[column].fillna(median)
        validation_values[column] = validation_values[column].fillna(median)

        raw_auc = roc_auc_score(validation[TARGET_COLUMN], validation_values[column])
        direction_agnostic_auc = max(raw_auc, 1.0 - raw_auc)
        stump = DecisionTreeClassifier(
            max_depth=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        stump.fit(train_values, train[TARGET_COLUMN])
        predicted = stump.predict(validation_values)
        stump_f1 = f1_score(
            validation[TARGET_COLUMN], predicted, average="macro", zero_division=0
        )
        direct_copy = bool(
            train_values[column].equals(train[TARGET_COLUMN])
            or train_values[column].equals(1 - train[TARGET_COLUMN])
        )
        result_rows.append(
            {
                "feature": column,
                "validation_direction_agnostic_roc_auc": direction_agnostic_auc,
                "validation_stump_macro_f1": stump_f1,
                "training_unique_values": train_values[column].nunique(),
                "direct_target_copy_on_train": direct_copy,
                "requires_investigation": direct_copy
                or direction_agnostic_auc >= UNIVARIATE_AUC_FLAG_THRESHOLD,
                "feature_policy": (
                    "quarantined"
                    if column in QUARANTINED_FEATURES
                    else "dataset_experiment_only"
                ),
            }
        )
    return pd.DataFrame(result_rows).sort_values(
        ["validation_direction_agnostic_roc_auc", "validation_stump_macro_f1"],
        ascending=False,
    )


def build_feature_provenance(columns: list[str]) -> pd.DataFrame:
    """Record the Week 6 handling decision for every raw column."""
    rows = []
    for column in columns:
        if column == TARGET_COLUMN:
            role = "target"
            decision = "exclude_from_features"
            reason = "Outcome label."
        elif column in IDENTIFIER_COLUMNS:
            role = "identifier"
            decision = "exclude_from_features"
            reason = "Identifier or raw text; retained only for grouping/future text work."
        elif column in QUARANTINED_FEATURES:
            role = "derived_feature"
            decision = "quarantine"
            reason = "Derived corpus/similarity provenance is not fold-safe enough for primary results."
        elif column == "TLD":
            role = "categorical_feature"
            decision = "retain_train_fitted_one_hot"
            reason = "Retain raw category; never use placeholder ordinal magnitude."
        else:
            role = "tabular_feature"
            decision = "retain_dataset_experiment_only"
            reason = "Valid for dataset comparison; deployment-time extraction is not yet audited."
        rows.append(
            {
                "feature": column,
                "role": role,
                "week6_decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Format selected columns as a compact Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    raw_rows: int,
    removed_duplicates: int,
    split_summary: pd.DataFrame,
    legacy_overlap: pd.DataFrame,
    univariate_results: pd.DataFrame,
) -> None:
    """Write the leakage-audit decision and evidence report."""
    flagged = univariate_results[univariate_results["requires_investigation"]]
    summary_view = split_summary[
        ["split", "rows", "domains", "class", "class_count", "class_percentage"]
    ]
    overlap_view = legacy_overlap[
        [
            "entity",
            "left_split",
            "right_split",
            "shared_unique_values",
            "right_rows_with_overlap",
            "right_overlap_percentage",
        ]
    ]
    univariate_view = univariate_results.head(12)[
        [
            "feature",
            "validation_direction_agnostic_roc_auc",
            "validation_stump_macro_f1",
            "requires_investigation",
            "feature_policy",
        ]
    ]
    lines = [
        "# Wider Target-Proxy and Leakage Audit",
        "",
        "## Decision",
        "",
        "The legacy row-random split is retained only as historical Week 5 evidence. "
        "Week 6 uses a new normalized-host-group-disjoint 70/15/15 protocol. Model and "
        "feature choices must use training and validation only; test remains frozen "
        "until each model configuration is locked.",
        "",
        "## Data Integrity",
        "",
        f"- Raw rows: {raw_rows:,}.",
        f"- Exact duplicate rows removed before splitting: {removed_duplicates:,}.",
        "- New split asserts zero exact URL and normalized-host overlap.",
        "- Split membership and source-file SHA-256 hashes are recorded in "
        "`data/processed/audited/split_manifest.json`.",
        "- Host grouping does not collapse sibling subdomains to registered "
        "domains; that requires a pinned Public Suffix List and remains a limitation.",
        "",
        "## Audited Split Summary",
        "",
        markdown_table(summary_view, summary_view.columns.tolist()),
        "",
        "## Legacy Split Overlap",
        "",
        markdown_table(overlap_view, overlap_view.columns.tolist()),
        "",
        "## Feature Policy",
        "",
        "- Excluded identifiers/raw text: `FILENAME`, `URL`, `Domain`, `Title`.",
        "- Quarantined from primary Week 6 modelling: "
        + ", ".join(f"`{name}`" for name in sorted(QUARANTINED_FEATURES))
        + ".",
        "- `TLD` is retained as raw categorical data and encoded by a train-fitted "
        "one-hot encoder.",
        "- Remaining webpage and URL-derived measurements are valid only for this "
        "dataset experiment until deployment-time extraction is separately audited.",
        "",
        "## Univariate Validation Audit",
        "",
        f"Features with direction-agnostic validation ROC-AUC >= "
        f"{UNIVARIATE_AUC_FLAG_THRESHOLD:.2f} are flagged for investigation, not "
        "automatically called leakage.",
        "",
        markdown_table(univariate_view, univariate_view.columns.tolist()),
        "",
        f"Flagged features: {', '.join(flagged['feature']) if len(flagged) else 'none'}.",
        "",
        "## Interpretation",
        "",
        "- No direct numeric copy of the target was found if the direct-copy column "
        "is false for every feature in the CSV evidence.",
        "- Strong single-feature separation can reflect dataset construction bias, "
        "a legitimate phishing indicator, or a target proxy. Statistics alone cannot "
        "distinguish those causes.",
        "- The new protocol removes normalized-host entity overlap and fails closed on "
        "the derived features with unresolved fold-safe provenance.",
        "- Legacy and audited scores are not directly comparable because the split and "
        "feature policy changed.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Generate audited splits and the wider leakage evidence package."""
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: Missing raw dataset: {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    print("Wider target-proxy and leakage audit")
    print("[1/6] Loading raw dataset and removing exact duplicate rows")
    raw = pd.read_csv(RAW_DATA_PATH)
    duplicate_count = int(raw.duplicated().sum())
    cleaned = raw.drop_duplicates().reset_index(drop=True)

    print("[2/6] Reconstructing legacy split overlap evidence")
    legacy_splits = reconstruct_legacy_splits(cleaned)
    legacy_overlap = build_overlap_table(legacy_splits)

    print("[3/6] Creating normalized-host-group-disjoint audited splits")
    audited_splits = build_audited_splits(cleaned)

    print("[4/6] Running train/validation-only univariate audit")
    univariate_results = audit_univariate_features(
        audited_splits["train"], audited_splits["validation"]
    )
    split_summary = build_split_summary(audited_splits)
    provenance = build_feature_provenance(cleaned.columns.tolist())

    print("[5/6] Saving audited data and evidence")
    AUDITED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split in audited_splits.items():
        split.to_csv(AUDITED_SPLIT_PATHS[split_name], index=False)
    write_split_manifest(audited_splits)
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False, float_format="%.6f")
    legacy_overlap.to_csv(LEGACY_OVERLAP_PATH, index=False, float_format="%.6f")
    univariate_results.to_csv(
        UNIVARIATE_RESULTS_PATH, index=False, float_format="%.6f"
    )
    provenance.to_csv(PROVENANCE_PATH, index=False)

    print("[6/6] Writing leakage audit report")
    write_report(
        raw_rows=len(raw),
        removed_duplicates=duplicate_count,
        split_summary=split_summary,
        legacy_overlap=legacy_overlap,
        univariate_results=univariate_results,
    )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Audited splits: {AUDITED_DATA_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
