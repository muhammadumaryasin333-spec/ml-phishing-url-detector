"""Basic exploration of the PhiUSIIL Phishing URL Dataset.

Week 2: dataset setup and exploration only. No model training.

Loads the raw CSV, prints a quick overview (shape, columns, dtypes,
missing values, duplicates, class balance) and writes two artifacts to
``reports/``:

* ``dataset_summary.txt`` - text summary of the dataset
* ``class_distribution.png`` - bar chart of the target class balance

Run from the project root:

    python src/data_exploration.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt
import pandas as pd

# Resolve paths relative to the project root (this file lives in src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUMMARY_PATH = REPORTS_DIR / "dataset_summary.txt"
CHART_PATH = REPORTS_DIR / "class_distribution.png"

# Candidate names for the target column, in priority order.
TARGET_CANDIDATES = ["label", "Label", "class", "Class", "result", "Result"]


def find_target_column(df: pd.DataFrame) -> str | None:
    """Return the most likely target column name, or None if not found."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    # Fall back to a case-insensitive match.
    lower_map = {c.lower(): c for c in df.columns}
    for name in TARGET_CANDIDATES:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def main() -> None:
    if not DATA_PATH.exists():
        print("=" * 70)
        print("Dataset file not found:")
        print(f"  {DATA_PATH}")
        print()
        print("Please download the PhiUSIIL Phishing URL Dataset from the")
        print("UCI Machine Learning Repository and place the CSV file at:")
        print("  data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
        print("=" * 70)
        return

    # Make sure the reports folder exists before we write anything.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # --- Console overview -------------------------------------------------
    print("\nShape (rows, columns):", df.shape)

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)
    print("Total missing values:", int(missing.sum()))

    duplicate_count = int(df.duplicated().sum())
    print("\nDuplicate rows:", duplicate_count)

    target = find_target_column(df)
    class_counts = None
    if target is None:
        print("\nCould not detect a target column.")
        print("Looked for:", ", ".join(TARGET_CANDIDATES))
    else:
        print(f"\nDetected target column: '{target}'")
        class_counts = df[target].value_counts(dropna=False)
        print("\nClass distribution:")
        print(class_counts)
        print("\nClass distribution (%):")
        print((class_counts / len(df) * 100).round(2))

    # --- Write text summary ----------------------------------------------
    write_summary(df, target, missing, duplicate_count, class_counts)
    print(f"\nSaved dataset summary to: {SUMMARY_PATH}")

    # --- Write class distribution chart ----------------------------------
    if class_counts is not None:
        save_class_chart(class_counts, target)
        print(f"Saved class distribution chart to: {CHART_PATH}")
    else:
        print("Skipped class distribution chart (no target column).")


def write_summary(df, target, missing, duplicate_count, class_counts) -> None:
    lines = []
    lines.append("PhiUSIIL Phishing URL Dataset - Summary")
    lines.append("=" * 50)
    lines.append(f"Source file: {DATA_PATH}")
    lines.append(f"Rows: {df.shape[0]}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append("")
    lines.append("Column names:")
    for col in df.columns:
        lines.append(f"  - {col} ({df[col].dtype})")
    lines.append("")
    lines.append(f"Total missing values: {int(missing.sum())}")
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        lines.append("Columns with missing values:")
        for col, n in cols_with_missing.items():
            lines.append(f"  - {col}: {int(n)}")
    else:
        lines.append("Columns with missing values: none")
    lines.append("")
    lines.append(f"Duplicate rows: {duplicate_count}")
    lines.append("")
    if target is not None and class_counts is not None:
        lines.append(f"Target column: {target}")
        lines.append("Class distribution:")
        for value, count in class_counts.items():
            pct = count / len(df) * 100
            lines.append(f"  - {value}: {int(count)} ({pct:.2f}%)")
    else:
        lines.append("Target column: not detected")
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines))


def save_class_chart(class_counts, target) -> None:
    labels = [str(v) for v in class_counts.index]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, class_counts.values, color=["#4c72b0", "#dd8452"][: len(labels)])
    plt.title("Class Distribution")
    plt.xlabel(f"{target} (class)")
    plt.ylabel("Count")
    for i, count in enumerate(class_counts.values):
        plt.text(i, count, str(int(count)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
