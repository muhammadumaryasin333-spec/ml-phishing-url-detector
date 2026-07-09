"""Preprocessing preparation for the PhiUSIIL Phishing URL Dataset.

Pre-IPR scope only: clean the tabular dataset and create reproducible
train/validation/test CSV files for later modelling. This script does not
train a model, scale features, create embeddings, or write model artifacts.

Run from the project root:

    python src/preprocess_data.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "reports" / "eda"
TABLES_DIR = EDA_DIR / "tables"

CLEANED_DATASET_PATH = PROCESSED_DIR / "cleaned_dataset.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
PREPROCESSING_REPORT_PATH = EDA_DIR / "preprocessing_report.txt"
SPLIT_DISTRIBUTION_PATH = TABLES_DIR / "split_distribution.csv"

TARGET_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]
ONE_HOT_MAX_UNIQUE_VALUES = 20
RANDOM_STATE = 42


def create_output_folders() -> None:
    """Create folders used by preprocessing outputs."""
    for folder in [PROCESSED_DIR, EDA_DIR, TABLES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load the raw dataset from the expected project path."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset file not found. Expected: "
            "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
        )
    print(f"[1/6] Loading dataset: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def find_target_column(df: pd.DataFrame) -> str | None:
    """Return the target column if it matches a known label name."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name

    lower_to_original = {column.lower(): column for column in df.columns}
    for name in TARGET_CANDIDATES:
        match = lower_to_original.get(name.lower())
        if match is not None:
            return match
    return None


def get_numeric_fill_values(df: pd.DataFrame, target_column: str) -> dict[str, float]:
    """Fit median values from one dataset, normally the training split."""
    feature_columns = [column for column in df.columns if column != target_column]
    numeric_columns = df[feature_columns].select_dtypes(include="number").columns
    return {column: df[column].median() for column in numeric_columns}


def fill_missing_values(
    df: pd.DataFrame,
    target_column: str,
    numeric_fill_values: dict[str, float],
) -> pd.DataFrame:
    """Fill missing values with simple, train-fitted tabular rules."""
    cleaned = df.copy()
    feature_columns = [column for column in cleaned.columns if column != target_column]
    categorical_columns = cleaned[feature_columns].select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column, median_value in numeric_fill_values.items():
        cleaned[column] = cleaned[column].fillna(median_value)

    for column in categorical_columns:
        cleaned[column] = cleaned[column].fillna("unknown")

    return cleaned


def fit_categorical_encoders(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[list[dict[str, object]], list[str]]:
    """Fit simple categorical encoders from one dataset, normally train only."""
    encoder_specs: list[dict[str, object]] = []
    encoding_notes = []
    categorical_columns = df.drop(columns=[target_column]).select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        values = df[column].astype("string").fillna("unknown")
        unique_count = int(values.nunique(dropna=False))
        if unique_count <= ONE_HOT_MAX_UNIQUE_VALUES:
            dummies = pd.get_dummies(values, prefix=column, dummy_na=False, dtype=int)
            encoder_specs.append(
                {
                    "column": column,
                    "method": "one_hot",
                    "dummy_columns": dummies.columns.tolist(),
                }
            )
            encoding_notes.append(
                f"{column}: one-hot encoded ({unique_count} unique values)"
            )
        else:
            categories = sorted(values.unique().tolist())
            mapping = {value: index for index, value in enumerate(categories)}
            encoder_specs.append(
                {
                    "column": column,
                    "method": "label",
                    "mapping": mapping,
                }
            )
            encoding_notes.append(
                f"{column}: label encoded from training split "
                f"({unique_count} training values, high cardinality)"
            )

    return encoder_specs, encoding_notes


def apply_categorical_encoders(
    df: pd.DataFrame,
    target_column: str,
    encoder_specs: list[dict[str, object]],
) -> pd.DataFrame:
    """Apply train-fitted categorical encoders to any split without refitting."""
    encoded = df.copy()
    for spec in encoder_specs:
        column = str(spec["column"])
        values = encoded[column].astype("string").fillna("unknown")

        if spec["method"] == "one_hot":
            dummy_columns = list(spec["dummy_columns"])
            dummies = pd.get_dummies(values, prefix=column, dummy_na=False, dtype=int)
            dummies = dummies.reindex(columns=dummy_columns, fill_value=0)
            encoded = pd.concat([encoded.drop(columns=[column]), dummies], axis=1)
        else:
            mapping = spec["mapping"]
            encoded[column] = values.map(mapping).fillna(-1).astype(int)

    return encoded


def build_split_distribution(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Create class distribution rows for each processed split."""
    rows = []
    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        counts = split_df[target_column].value_counts(dropna=False).sort_index()
        for label_value, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "class": label_value,
                    "count": int(count),
                    "percentage": round(count / len(split_df) * 100, 2),
                }
            )
    return pd.DataFrame(rows)


def write_preprocessing_report(
    raw_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    duplicate_count: int,
    target_column: str,
    cleaned_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_distribution: pd.DataFrame,
    encoding_notes: list[str],
) -> None:
    """Write a concise preprocessing report for IPR evidence."""
    lines = [
        "PhiUSIIL Phishing URL Dataset - Preprocessing Report",
        "=" * 60,
        f"Raw dataset shape: {raw_shape}",
        f"Cleaned dataset shape: {cleaned_shape}",
        f"Number of duplicates removed: {duplicate_count:,}",
        "",
        "Missing value handling method:",
        "  - Numeric feature columns: filled with training-split median",
        '  - Categorical/object feature columns: filled with "unknown"',
        "  - Target label kept unchanged",
        "",
        f"Target column used: {target_column}",
        "Class meaning:",
        "  - 1 = legitimate",
        "  - 0 = phishing",
        "",
        "Final class distribution:",
    ]

    final_counts = cleaned_df[target_column].value_counts(dropna=False).sort_index()
    for label_value, count in final_counts.items():
        lines.append(
            f"  - {label_value}: {int(count):,} rows ({count / len(cleaned_df) * 100:.2f}%)"
        )

    lines.extend(
        [
            "",
            "Encoding method:",
            "  - Encoders are fitted on the training split only to avoid validation/test vocabulary leakage",
            f"  - One-hot encoding used for training categorical features with <= {ONE_HOT_MAX_UNIQUE_VALUES} unique values",
            "  - Label encoding used for higher-cardinality categorical/text features as a simple placeholder",
            "  - Unseen validation/test categorical values are encoded as -1",
        ]
    )
    if encoding_notes:
        for note in encoding_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  - No categorical/object feature columns required encoding")

    lines.extend(
        [
            "",
            f"Train size: {len(train_df):,} rows",
            f"Validation size: {len(validation_df):,} rows",
            f"Test size: {len(test_df):,} rows",
            "",
            "Train/validation/test class distributions:",
        ]
    )

    for split_name in ["train", "validation", "test"]:
        split_rows = split_distribution[split_distribution["split"] == split_name]
        lines.append(f"  - {split_name}:")
        for _, row in split_rows.iterrows():
            lines.append(
                f"      class {row['class']}: {int(row['count']):,} rows ({row['percentage']:.2f}%)"
            )

    lines.extend(
        [
            "",
            "Output file paths:",
            f"  - Cleaned dataset: {CLEANED_DATASET_PATH}",
            f"  - Train split: {TRAIN_PATH}",
            f"  - Validation split: {VALIDATION_PATH}",
            f"  - Test split: {TEST_PATH}",
            f"  - Split distribution table: {SPLIT_DISTRIBUTION_PATH}",
            "",
            "Preprocessing note:",
            "  No scaling, model training, SHAP explanations, transformer embeddings, or model files were created.",
            "  High-cardinality label encoding is for pre-IPR dataset preparation only; later modelling should replace it with a fitted sklearn preprocessing pipeline or text/URL feature extraction.",
        ]
    )

    PREPROCESSING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    create_output_folders()

    try:
        raw_df = load_dataset()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return

    target_column = find_target_column(raw_df)
    if target_column is None:
        print("ERROR: Could not detect target column.")
        print(f"Looked for: {', '.join(TARGET_CANDIDATES)}")
        return

    print(f"Detected target column: {target_column}")
    print("Class meaning: 1 = legitimate, 0 = phishing")

    print("[2/6] Removing exact duplicate rows if present")
    duplicate_count = int(raw_df.duplicated().sum())
    cleaned_df = raw_df.drop_duplicates().reset_index(drop=True)

    print("[3/6] Creating stratified train/validation/test splits")
    train_df, temp_df = train_test_split(
        cleaned_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=cleaned_df[target_column],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[target_column],
    )

    print("[4/6] Filling missing values with training-split medians")
    numeric_fill_values = get_numeric_fill_values(train_df, target_column)
    train_df = fill_missing_values(train_df, target_column, numeric_fill_values)
    validation_df = fill_missing_values(validation_df, target_column, numeric_fill_values)
    test_df = fill_missing_values(test_df, target_column, numeric_fill_values)
    cleaned_df = fill_missing_values(cleaned_df, target_column, numeric_fill_values)

    print("[5/6] Encoding categorical/object feature columns with train-fitted encoders")
    encoder_specs, encoding_notes = fit_categorical_encoders(train_df, target_column)
    train_df = apply_categorical_encoders(train_df, target_column, encoder_specs)
    validation_df = apply_categorical_encoders(validation_df, target_column, encoder_specs)
    test_df = apply_categorical_encoders(test_df, target_column, encoder_specs)
    cleaned_df = apply_categorical_encoders(cleaned_df, target_column, encoder_specs)

    split_distribution = build_split_distribution(
        train_df, validation_df, test_df, target_column
    )

    print("[6/6] Saving processed datasets and preprocessing report")
    cleaned_df.to_csv(CLEANED_DATASET_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    split_distribution.to_csv(SPLIT_DISTRIBUTION_PATH, index=False)

    write_preprocessing_report(
        raw_shape=raw_df.shape,
        cleaned_shape=cleaned_df.shape,
        duplicate_count=duplicate_count,
        target_column=target_column,
        cleaned_df=cleaned_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        split_distribution=split_distribution,
        encoding_notes=encoding_notes,
    )

    print("Preprocessing complete.")
    print(f"Cleaned dataset: {CLEANED_DATASET_PATH}")
    print(f"Train split: {TRAIN_PATH}")
    print(f"Validation split: {VALIDATION_PATH}")
    print(f"Test split: {TEST_PATH}")
    print(f"Report: {PREPROCESSING_REPORT_PATH}")


if __name__ == "__main__":
    main()
