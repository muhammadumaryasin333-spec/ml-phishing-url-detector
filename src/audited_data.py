"""Shared data protocol for leakage-audited model experiments."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
AUDITED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "audited"
AUDIT_REPORT_DIR = PROJECT_ROOT / "reports" / "model_results" / "leakage_audit"

AUDITED_SPLIT_PATHS = {
    "train": AUDITED_DATA_DIR / "train.csv",
    "validation": AUDITED_DATA_DIR / "validation.csv",
    "test": AUDITED_DATA_DIR / "test.csv",
}
SPLIT_MANIFEST_PATH = AUDITED_DATA_DIR / "split_manifest.json"

TARGET_COLUMN = "label"
GROUP_COLUMN = "Domain"
MEMBERSHIP_COLUMN = "FILENAME"
RANDOM_STATE = 42
N_SPLIT_FOLDS = 20
TRAIN_FOLDS = frozenset(range(14))
VALIDATION_FOLDS = frozenset(range(14, 17))
TEST_FOLDS = frozenset(range(17, 20))

IDENTIFIER_COLUMNS = frozenset({"FILENAME", "URL", "Domain", "Title"})
QUARANTINED_FEATURES = frozenset(
    {
        "URLSimilarityIndex",
        "CharContinuationRate",
        "TLDLegitimateProb",
        "URLCharProb",
    }
)


def normalize_hostname(value: object, *, value_is_url: bool = False) -> str:
    """Return a stable hostname group from a domain authority or full URL."""
    if pd.isna(value):
        raise ValueError("Hostname values must not be missing.")
    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("Hostname values must not be empty.")

    parsed = urlsplit(raw_value if value_is_url else f"//{raw_value}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Could not parse a hostname from {raw_value!r}.")
    hostname = hostname.casefold().rstrip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    if not hostname:
        raise ValueError(f"Could not normalize a hostname from {raw_value!r}.")

    try:
        return ipaddress.ip_address(hostname).compressed
    except ValueError:
        try:
            return hostname.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise ValueError(f"Invalid IDNA hostname: {raw_value!r}.") from error


def normalize_groups(values: pd.Series) -> pd.Series:
    """Normalize domain-authority strings used as split groups."""
    return values.map(normalize_hostname).astype("string")


def normalize_url_hosts(values: pd.Series) -> pd.Series:
    """Normalize hostnames parsed independently from full URL strings."""
    return values.map(lambda value: normalize_hostname(value, value_is_url=True)).astype(
        "string"
    )


def assign_group_folds(df: pd.DataFrame) -> pd.Series:
    """Assign every row to one deterministic stratified domain-group fold."""
    required_columns = {TARGET_COLUMN, GROUP_COLUMN, "URL"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot create audited splits; missing columns: "
            + ", ".join(missing_columns)
        )
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target labels must not contain missing values.")
    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Audited splitting requires exactly two target classes.")

    groups = normalize_groups(df[GROUP_COLUMN])
    url_hosts = normalize_url_hosts(df["URL"])
    mismatched_hosts = groups.ne(url_hosts)
    if mismatched_hosts.any():
        raise ValueError(
            "Domain and URL hostnames disagree for "
            f"{int(mismatched_hosts.sum())} rows."
        )
    if groups.nunique() < N_SPLIT_FOLDS:
        raise ValueError(
            f"At least {N_SPLIT_FOLDS} distinct domain groups are required."
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLIT_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_assignments = pd.Series(-1, index=df.index, dtype="int16")
    placeholder_features = pd.DataFrame(index=df.index)
    for fold_index, (_, fold_rows) in enumerate(
        splitter.split(placeholder_features, df[TARGET_COLUMN], groups)
    ):
        fold_assignments.iloc[fold_rows] = fold_index

    if fold_assignments.lt(0).any():
        raise RuntimeError("Some rows were not assigned to an audited fold.")
    return fold_assignments


def build_audited_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 70/15/15 host-group-disjoint splits from cleaned raw rows."""
    if MEMBERSHIP_COLUMN not in df:
        raise ValueError(f"Missing membership column: {MEMBERSHIP_COLUMN}")
    if df[MEMBERSHIP_COLUMN].astype("string").duplicated().any():
        raise ValueError("Membership identifiers must be unique before splitting.")
    ordered = df.sort_values(MEMBERSHIP_COLUMN, kind="stable").reset_index(drop=True)
    folds = assign_group_folds(ordered)
    fold_sets = {
        "train": TRAIN_FOLDS,
        "validation": VALIDATION_FOLDS,
        "test": TEST_FOLDS,
    }
    splits = {
        split_name: ordered.loc[folds.isin(split_folds)].reset_index(drop=True)
        for split_name, split_folds in fold_sets.items()
    }
    validate_audited_splits(splits)
    return splits


def validate_audited_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Reject incomplete, schema-mismatched, or entity-overlapping splits."""
    expected_names = {"train", "validation", "test"}
    if set(splits) != expected_names:
        raise ValueError("Audited splits must contain train, validation, and test.")
    if any(split.empty for split in splits.values()):
        raise ValueError("Audited splits must not be empty.")

    train_columns = splits["train"].columns.tolist()
    for split_name, split in splits.items():
        if split.columns.tolist() != train_columns:
            raise ValueError(f"Feature schema differs in {split_name} split.")
        if set(split[TARGET_COLUMN].unique()) != {0, 1}:
            raise ValueError(f"Both target classes are required in {split_name} split.")

    normalized_domains = {
        name: set(normalize_groups(split[GROUP_COLUMN]))
        for name, split in splits.items()
    }
    urls = {
        name: set(split["URL"].astype("string")) for name, split in splits.items()
    }
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        if normalized_domains[left] & normalized_domains[right]:
            raise ValueError(f"Domain overlap detected between {left} and {right}.")
        if urls[left] & urls[right]:
            raise ValueError(f"URL overlap detected between {left} and {right}.")

    memberships = [
        split[MEMBERSHIP_COLUMN].astype("string") for split in splits.values()
    ]
    combined_memberships = pd.concat(memberships, ignore_index=True)
    if combined_memberships.duplicated().any():
        raise ValueError("Membership identifiers overlap or repeat across audited splits.")

    combined_targets = pd.concat(
        [split[TARGET_COLUMN] for split in splits.values()], ignore_index=True
    )
    global_positive_rate = float(combined_targets.mean())
    expected_shares = {"train": 0.70, "validation": 0.15, "test": 0.15}
    total_rows = len(combined_targets)
    for split_name, split in splits.items():
        row_share = len(split) / total_rows
        if abs(row_share - expected_shares[split_name]) > 0.01:
            raise ValueError(f"Row share is outside tolerance for {split_name} split.")
        if abs(float(split[TARGET_COLUMN].mean()) - global_positive_rate) > 0.01:
            raise ValueError(
                f"Class prevalence is outside tolerance for {split_name} split."
            )


def prepare_model_data(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Return model features and targets under the locked audited feature policy."""
    validate_audited_splits(splits)
    excluded_columns = IDENTIFIER_COLUMNS | QUARANTINED_FEATURES | {TARGET_COLUMN}
    features = {
        name: split.drop(columns=sorted(excluded_columns)).copy()
        for name, split in splits.items()
    }
    targets = {name: split[TARGET_COLUMN].copy() for name, split in splits.items()}

    train_columns = features["train"].columns.tolist()
    if not train_columns:
        raise ValueError("No model features remain after audited exclusions.")
    for split_name in ["validation", "test"]:
        features[split_name] = features[split_name].reindex(columns=train_columns)
    return features, targets


def load_audited_splits() -> dict[str, pd.DataFrame]:
    """Load and validate the generated audited split files."""
    missing_paths = [path for path in AUDITED_SPLIT_PATHS.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing audited split files: "
            + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing_paths)
        )
    splits = {name: pd.read_csv(path) for name, path in AUDITED_SPLIT_PATHS.items()}
    validate_audited_splits(splits)
    return splits


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_hash(split: pd.DataFrame) -> str:
    """Hash sorted immutable row identifiers for one split."""
    if MEMBERSHIP_COLUMN not in split:
        raise ValueError(f"Missing membership column: {MEMBERSHIP_COLUMN}")
    members = sorted(split[MEMBERSHIP_COLUMN].astype("string").tolist())
    return hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()


def write_split_manifest(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Persist protocol configuration, membership hashes, and class counts."""
    manifest: dict[str, object] = {
        "protocol": "stratified_normalized_host_group_70_15_15",
        "random_state": RANDOM_STATE,
        "fold_count": N_SPLIT_FOLDS,
        "fold_assignment": {
            "train": sorted(TRAIN_FOLDS),
            "validation": sorted(VALIDATION_FOLDS),
            "test": sorted(TEST_FOLDS),
        },
        "group_column": GROUP_COLUMN,
        "group_normalization": (
            "urlsplit_hostname_casefold_strip_trailing_dot_idna_"
            "canonicalize_ip_remove_one_www_prefix"
        ),
        "source_path": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(RAW_DATA_PATH),
        "identifier_columns_excluded": sorted(IDENTIFIER_COLUMNS),
        "quarantined_features": sorted(QUARANTINED_FEATURES),
        "splits": {},
    }
    split_details = manifest["splits"]
    assert isinstance(split_details, dict)
    for name, split in splits.items():
        split_details[name] = {
            "rows": len(split),
            "domains": int(normalize_groups(split[GROUP_COLUMN]).nunique()),
            "class_counts": {
                str(label): int(count)
                for label, count in split[TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            },
            "membership_sha256": membership_hash(split),
        }
    SPLIT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
