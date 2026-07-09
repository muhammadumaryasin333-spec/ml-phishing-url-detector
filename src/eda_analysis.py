"""Complete EDA workflow for the PhiUSIIL Phishing URL Dataset.

Pre-IPR scope only: dataset understanding, charts, tables, and feature
selection planning notes. This script does not train any machine learning
model and does not remove rows or columns from the raw data.

Run from the project root:

    python src/eda_analysis.py
"""

from pathlib import Path
import os
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
EDA_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURES_DIR = EDA_DIR / "figures"
TABLES_DIR = EDA_DIR / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OVERVIEW_PATH = EDA_DIR / "dataset_overview.txt"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "phishing-url-detector-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a display.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]
HIGH_CARDINALITY_THRESHOLD = 50
MAX_PLOT_FEATURES = 12
MAX_HEATMAP_FEATURES = 25


def create_output_folders() -> None:
    """Create report and processed-data folders used by the workflow."""
    for folder in [EDA_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load the raw PhiUSIIL dataset from the expected project path."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset file not found. Expected: "
            "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
        )
    print(f"[1/8] Loading dataset: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def find_target_column(df: pd.DataFrame) -> str | None:
    """Return the target column name if it matches a known target pattern."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name

    lower_to_original = {column.lower(): column for column in df.columns}
    for name in TARGET_CANDIDATES:
        match = lower_to_original.get(name.lower())
        if match is not None:
            return match
    return None


def get_column_groups(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Group features with simple phishing-domain keyword rules."""
    group_keywords = {
        "URL structure features": [
            "url",
            "redirect",
            "ref",
            "subdomain",
            "https",
            "obfus",
        ],
        "Domain-related features": ["domain", "tld", "favicon", "robots"],
        "Character-based features": [
            "char",
            "letter",
            "digit",
            "degit",
            "equals",
            "qmark",
            "ampersand",
            "special",
            "spacial",
            "length",
            "ratio",
        ],
        "Webpage/source-code features": [
            "lineofcode",
            "largestline",
            "iframe",
            "form",
            "submit",
            "hidden",
            "password",
            "popup",
            "image",
            "css",
            "js",
            "responsive",
            "description",
            "copyright",
        ],
        "Content/title-related features": [
            "title",
            "social",
            "bank",
            "pay",
            "crypto",
        ],
    }

    rows = []
    for column in df.columns:
        if column == target_column:
            group = "target"
        else:
            normalized = column.lower()
            group = "Other numeric features"
            for candidate_group, keywords in group_keywords.items():
                if any(keyword in normalized for keyword in keywords):
                    group = candidate_group
                    break
            if group == "Other numeric features" and not pd.api.types.is_numeric_dtype(df[column]):
                group = "Other categorical/text features"

        rows.append(
            {
                "column": column,
                "feature_group": group,
                "dtype": str(df[column].dtype),
                "unique_values": int(df[column].nunique(dropna=False)),
            }
        )

    return pd.DataFrame(rows)


def save_basic_tables(
    df: pd.DataFrame,
    target_column: str,
    missing_table: pd.DataFrame,
    duplicate_count: int,
) -> pd.Series:
    """Save overview tables used by the text report."""
    print("[2/8] Saving overview tables")

    column_dtypes = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "unique_values": [df[column].nunique(dropna=False) for column in df.columns],
        }
    )
    column_dtypes.to_csv(TABLES_DIR / "column_dtypes.csv", index=False)
    missing_table.to_csv(TABLES_DIR / "missing_values.csv", index=False)

    duplicate_summary = pd.DataFrame(
        [{"duplicate_rows": duplicate_count, "duplicate_percentage": duplicate_count / len(df) * 100}]
    )
    duplicate_summary.to_csv(TABLES_DIR / "duplicate_summary.csv", index=False)

    class_counts = df[target_column].value_counts(dropna=False).sort_index()
    class_distribution = pd.DataFrame(
        {
            "class": class_counts.index,
            "count": class_counts.values,
            "percentage": (class_counts.values / len(df) * 100).round(2),
            "meaning": [class_meaning(value) for value in class_counts.index],
        }
    )
    class_distribution.to_csv(TABLES_DIR / "class_distribution.csv", index=False)
    return class_counts


def class_meaning(value: object) -> str:
    """Document the class labels used by the PhiUSIIL dataset."""
    if value == 1 or str(value) == "1":
        return "legitimate"
    if value == 0 or str(value) == "0":
        return "phishing"
    return "unknown"


def build_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value summary for every column."""
    missing_count = df.isna().sum()
    return pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percentage": (missing_count.values / len(df) * 100).round(4),
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])


def save_class_distribution_plots(class_counts: pd.Series, target_column: str) -> None:
    """Save count and percentage charts for the target distribution."""
    print("[3/8] Saving class distribution charts")
    labels = [f"{value} ({class_meaning(value)})" for value in class_counts.index]
    colors = ["#c94f4f", "#2f7d62", "#4c72b0", "#8172b2"][: len(labels)]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=class_counts.values, hue=labels, palette=colors, legend=False)
    plt.title("Class Distribution - Count")
    plt.xlabel(target_column)
    plt.ylabel("Rows")
    for index, count in enumerate(class_counts.values):
        plt.text(index, count, f"{int(count):,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_count.png", dpi=160)
    plt.close()

    percentages = class_counts / class_counts.sum() * 100
    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=percentages.values, hue=labels, palette=colors, legend=False)
    plt.title("Class Distribution - Percentage")
    plt.xlabel(target_column)
    plt.ylabel("Percentage")
    plt.ylim(0, max(100, percentages.max() + 10))
    for index, percentage in enumerate(percentages.values):
        plt.text(index, percentage, f"{percentage:.2f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_percentage.png", dpi=160)
    plt.close()


def save_missing_plot(missing_table: pd.DataFrame) -> None:
    """Save a missing-value bar chart only when missing values exist."""
    missing_columns = missing_table[missing_table["missing_count"] > 0].head(20)
    if missing_columns.empty:
        print("[4/8] No missing values found; skipping missing-values chart")
        return

    print("[4/8] Saving missing-values chart")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=missing_columns, y="column", x="missing_count", color="#4c72b0")
    plt.title("Top Columns With Missing Values")
    plt.xlabel("Missing values")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values_bar.png", dpi=160)
    plt.close()


def get_selected_numeric_features(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
    limit: int,
) -> list[str]:
    """Select readable chart features using target correlation and variance."""
    feature_columns = [column for column in numeric_columns if column != target_column]
    if not feature_columns:
        return []

    selected: list[str] = []
    if target_column in numeric_columns:
        correlations = (
            df[feature_columns]
            .corrwith(df[target_column], numeric_only=True)
            .dropna()
            .abs()
            .sort_values(ascending=False)
        )
        selected.extend(correlations.head(limit).index.tolist())

    if len(selected) < limit:
        variances = df[feature_columns].var(numeric_only=True).sort_values(ascending=False)
        for column in variances.index:
            if column not in selected:
                selected.append(column)
            if len(selected) >= limit:
                break

    return selected[:limit]


def save_numeric_analysis(df: pd.DataFrame, target_column: str) -> None:
    """Save numeric statistics, correlations, heatmap, distributions, and outliers."""
    print("[5/8] Saving numeric analysis tables and charts")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    numeric_df = df[numeric_columns]

    numeric_df.describe().T.to_csv(TABLES_DIR / "numeric_descriptive_statistics.csv")

    correlation_matrix = numeric_df.corr()
    correlation_matrix.to_csv(TABLES_DIR / "correlation_matrix.csv")
    save_highly_correlated_pairs(correlation_matrix, target_column)
    save_outlier_summary(df, numeric_columns)

    selected_features = get_selected_numeric_features(
        df, numeric_columns, target_column, MAX_PLOT_FEATURES
    )
    if selected_features:
        plot_numeric_distributions(df, selected_features)
        plot_numeric_boxplots(df, selected_features)

    heatmap_features = get_selected_numeric_features(
        df, numeric_columns, target_column, MAX_HEATMAP_FEATURES
    )
    if target_column in numeric_columns and target_column not in heatmap_features:
        heatmap_features = [target_column] + heatmap_features
    plot_correlation_heatmap(df, heatmap_features)
    plot_target_correlations(df, numeric_columns, target_column)


def save_highly_correlated_pairs(correlation_matrix: pd.DataFrame, target_column: str) -> None:
    """Save feature pairs with absolute correlation of 0.90 or higher."""
    rows = []
    columns = correlation_matrix.columns.tolist()
    for left_index, left_column in enumerate(columns):
        for right_column in columns[left_index + 1 :]:
            if target_column in {left_column, right_column}:
                continue
            correlation = correlation_matrix.loc[left_column, right_column]
            if pd.notna(correlation) and abs(correlation) >= 0.90:
                rows.append(
                    {
                        "feature_1": left_column,
                        "feature_2": right_column,
                        "correlation": round(float(correlation), 6),
                        "absolute_correlation": round(abs(float(correlation)), 6),
                    }
                )

    columns = ["feature_1", "feature_2", "correlation", "absolute_correlation"]
    highly_correlated = pd.DataFrame(rows, columns=columns)
    if not highly_correlated.empty:
        highly_correlated = highly_correlated.sort_values(
            "absolute_correlation", ascending=False
        )
    highly_correlated.to_csv(TABLES_DIR / "highly_correlated_features.csv", index=False)


def save_outlier_summary(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """Use IQR rules to summarize possible outliers without removing them."""
    rows = []
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            lower_bound = upper_bound = None
            outlier_count = 0
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        rows.append(
            {
                "column": column,
                "outlier_count": outlier_count,
                "outlier_percentage": round(outlier_count / len(df) * 100, 4),
                "iqr_lower_bound": lower_bound,
                "iqr_upper_bound": upper_bound,
            }
        )

    pd.DataFrame(rows).sort_values(
        ["outlier_count", "column"], ascending=[False, True]
    ).to_csv(TABLES_DIR / "outlier_summary.csv", index=False)


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str]) -> None:
    """Plot histograms for selected numeric features."""
    plot_df = df[columns]
    axes = plot_df.hist(bins=30, figsize=(14, 10), color="#4c72b0", edgecolor="white")
    for axis in axes.flatten():
        axis.set_title(axis.get_title(), fontsize=9)
    plt.suptitle("Selected Numeric Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=160)
    plt.close()


def plot_numeric_boxplots(df: pd.DataFrame, columns: list[str]) -> None:
    """Plot boxplots for selected numeric features to inspect outliers."""
    plot_df = df[columns].melt(var_name="feature", value_name="value")
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=plot_df, x="feature", y="value", color="#72a0c1", showfliers=True)
    plt.title("Selected Numeric Feature Boxplots")
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_boxplots.png", dpi=160)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> None:
    """Plot a readable correlation heatmap for selected numeric features."""
    if len(columns) < 2:
        return

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        df[columns].corr(),
        cmap="vlag",
        center=0,
        linewidths=0.3,
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap - Selected Numeric Features")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()


def plot_target_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
) -> None:
    """Save a bar chart of strongest numeric feature correlations with target."""
    if target_column not in numeric_columns:
        return

    feature_columns = [column for column in numeric_columns if column != target_column]
    correlations = (
        df[feature_columns]
        .corrwith(df[target_column], numeric_only=True)
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(20)
    )
    if correlations.empty:
        return

    plt.figure(figsize=(10, 8))
    colors = ["#2f7d62" if value >= 0 else "#c94f4f" for value in correlations.values]
    sns.barplot(x=correlations.values, y=correlations.index, hue=correlations.index, palette=colors, legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Strongest Feature Correlations With {target_column}")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_correlation_bar.png", dpi=160)
    plt.close()


def save_categorical_analysis(df: pd.DataFrame) -> None:
    """Save categorical summary without plotting high-cardinality text fields."""
    print("[6/8] Saving categorical/text feature summary")
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    rows = []

    for column in categorical_columns:
        value_counts = df[column].value_counts(dropna=False).head(10)
        top_values = "; ".join(
            f"{str(value)[:80]} ({int(count)})" for value, count in value_counts.items()
        )
        missing_count = int(df[column].isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "unique_values": int(df[column].nunique(dropna=False)),
                "missing_count": missing_count,
                "missing_percentage": round(missing_count / len(df) * 100, 4),
                "top_10_values": top_values,
                "high_cardinality": df[column].nunique(dropna=False)
                > HIGH_CARDINALITY_THRESHOLD,
            }
        )

    pd.DataFrame(rows).to_csv(TABLES_DIR / "categorical_summary.csv", index=False)


def write_dataset_overview(
    df: pd.DataFrame,
    target_column: str,
    class_counts: pd.Series,
    missing_table: pd.DataFrame,
    duplicate_count: int,
    feature_groups: pd.DataFrame,
) -> None:
    """Write a readable text summary of the complete EDA findings."""
    print("[8/8] Writing dataset overview report")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    high_cardinality_columns = [
        column
        for column in categorical_columns
        if df[column].nunique(dropna=False) > HIGH_CARDINALITY_THRESHOLD
    ]

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    lines = [
        "PhiUSIIL Phishing URL Dataset - Complete EDA Overview",
        "=" * 60,
        f"Source file: {DATA_PATH}",
        f"Dataset shape: {df.shape}",
        f"Number of rows: {df.shape[0]:,}",
        f"Number of columns: {df.shape[1]:,}",
        f"Memory usage: {memory_mb:.2f} MB",
        "",
        "Column names:",
    ]
    lines.extend(f"  - {column}" for column in df.columns)

    lines.extend(["", "Data types:"])
    for column, dtype in df.dtypes.items():
        lines.append(f"  - {column}: {dtype}")

    lines.extend(["", "First few rows summary:"])
    lines.append(df.head().to_string(index=False))

    lines.extend(
        [
            "",
            f"Target column name: {target_column}",
            "Target class meaning for this dataset:",
            "  - 1 = legitimate",
            "  - 0 = phishing",
            "Target class distribution:",
        ]
    )
    for value, count in class_counts.items():
        percentage = count / len(df) * 100
        lines.append(
            f"  - {value} ({class_meaning(value)}): {int(count):,} rows ({percentage:.2f}%)"
        )

    lines.extend(
        [
            "",
            f"Duplicate row count: {duplicate_count:,}",
            "",
            "Missing value summary:",
        ]
    )
    total_missing = int(missing_table["missing_count"].sum())
    if total_missing == 0:
        lines.append("  - No missing values were found in the dataset.")
    else:
        for _, row in missing_table[missing_table["missing_count"] > 0].iterrows():
            lines.append(
                f"  - {row['column']}: {int(row['missing_count']):,} "
                f"({row['missing_percentage']:.4f}%)"
            )

    lines.extend(
        [
            "",
            f"Number of numeric columns: {len(numeric_columns):,}",
            f"Number of categorical/object columns: {len(categorical_columns):,}",
            "Constant columns:",
        ]
    )
    if constant_columns:
        lines.extend(f"  - {column}" for column in constant_columns)
    else:
        lines.append("  - None")

    lines.append("High-cardinality categorical/object columns:")
    if high_cardinality_columns:
        for column in high_cardinality_columns:
            lines.append(
                f"  - {column}: {df[column].nunique(dropna=False):,} unique values"
            )
    else:
        lines.append("  - None")

    lines.extend(["", "Feature group analysis:"])
    lines.append(
        "  Features were grouped with simple keyword rules based on column names. "
        "This is for EDA organization only; unknown or ambiguous columns are left in an other group."
    )
    group_counts = feature_groups["feature_group"].value_counts().sort_index()
    for group, count in group_counts.items():
        lines.append(f"  - {group}: {int(count)} columns")

    lines.extend(
        [
            "",
            "EDA note:",
            "  No rows or columns were removed by this EDA script. Highly correlated "
            "features and outliers are reported for later feature selection planning only.",
        ]
    )

    OVERVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    create_output_folders()

    try:
        df = load_dataset()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return

    target_column = find_target_column(df)
    if target_column is None:
        print("ERROR: Could not detect target column.")
        print(f"Looked for: {', '.join(TARGET_CANDIDATES)}")
        return

    print(f"Detected target column: {target_column}")
    print("Class meaning: 1 = legitimate, 0 = phishing")

    missing_table = build_missing_table(df)
    duplicate_count = int(df.duplicated().sum())
    class_counts = save_basic_tables(df, target_column, missing_table, duplicate_count)
    save_class_distribution_plots(class_counts, target_column)
    save_missing_plot(missing_table)
    save_numeric_analysis(df, target_column)
    save_categorical_analysis(df)

    print("[7/8] Saving feature grouping table")
    feature_groups = get_column_groups(df, target_column)
    feature_groups.to_csv(TABLES_DIR / "feature_groups.csv", index=False)

    write_dataset_overview(
        df,
        target_column,
        class_counts,
        missing_table,
        duplicate_count,
        feature_groups,
    )

    print("EDA complete.")
    print(f"Report folder: {EDA_DIR}")
    print(f"Figures folder: {FIGURES_DIR}")
    print(f"Tables folder: {TABLES_DIR}")


if __name__ == "__main__":
    main()
