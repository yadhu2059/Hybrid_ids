"""
Dataset Preparation, Cleaning, and Time-Based Splitting Pipeline
Implements strict temporal splits, deduplication, handling of Inf/NaN,
and standardization of known CIC-IDS2017 label quirks.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from app.ml.features import (
    FEATURE_NAMES,
    METADATA_NAMES,
    extract_features_batch,
    extract_features_from_flow
)

# Known CIC-IDS2017 Label Normalization Map
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    # Benign
    "benign": "Benign",
    "normal": "Benign",
    # DoS / DDoS
    "dos-loic-udp": "DoS-LOIC-UDP",
    "dos loic udp": "DoS-LOIC-UDP",
    "dos-loic-tcp": "DoS-LOIC-TCP",
    "dos loic tcp": "DoS-LOIC-TCP",
    "dos-slowloris": "DoS-Slowloris",
    "dos slowloris": "DoS-Slowloris",
    "dos-slowhttptest": "DoS-SlowHTTPTest",
    "dos slowhttptest": "DoS-SlowHTTPTest",
    "dos-hulk-goldeneye": "DoS-Hulk-GoldenEye",
    "dos hulk goldeneye": "DoS-Hulk-GoldenEye",
    "dos hulk": "DoS-Hulk-GoldenEye",
    "dos goldeneye": "DoS-Hulk-GoldenEye",
    "ddos": "DoS-LOIC-UDP",
    # Scans
    "portscan-tcp-syn": "PortScan-TCP-SYN",
    "portscan tcp syn": "PortScan-TCP-SYN",
    "portscan-connect": "PortScan-Connect",
    "portscan connect": "PortScan-Connect",
    "portscan": "PortScan-Connect",
    # Brute Force
    "bruteforce-ssh-patator": "BruteForce-SSH-Patator",
    "ssh-patator": "BruteForce-SSH-Patator",
    "bruteforce-ftp-patator": "BruteForce-FTP-Patator",
    "ftp-patator": "BruteForce-FTP-Patator",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack brute force": "Web Attack - Brute Force",
    # Web Attacks
    "web attack - sql injection": "Web Attack - SQL Injection",
    "web attack sql injection": "Web Attack - SQL Injection",
    "web attack - sqli": "Web Attack - SQL Injection",
    "webattack-sqli": "Web Attack - SQL Injection",
    "web attack - xss": "Web Attack - XSS",
    "web attack xss": "Web Attack - XSS",
    "webattack-xss": "Web Attack - XSS",
    # Botnet & Infiltration
    "botnet-c2-beaconing": "Botnet-C2-Beaconing",
    "bot": "Botnet-C2-Beaconing",
    "botnet": "Botnet-C2-Beaconing",
    "infiltration": "Infiltration",
    "infiltration-lateralsweep": "Infiltration",
    "heartbleed": "DoS-Slowloris"
}


def normalize_label(label: Any) -> str:
    """Standardizes dataset labels into canonical attack and benign categories."""
    if label is None or pd.isna(label):
        return "Benign"
    clean = str(label).strip()
    key = clean.lower()
    return LABEL_NORMALIZATION_MAP.get(key, clean)


def clean_dataset(
    df: pd.DataFrame,
    dedup_cols: Optional[List[str]] = None,
    drop_unlabeled: bool = True
) -> pd.DataFrame:
    """
    Cleans raw network flow dataset:
    1. Trims whitespace from column names.
    2. Drops duplicate flows.
    3. Replaces inf, -inf, and NaN with bounded clean numbers.
    4. Normalizes labels according to CIC-IDS2017 standard taxonomy.
    5. Filters invalid flow durations (< 0).
    """
    df = df.copy()
    # Strip whitespace from column names (CIC-IDS2017 often has ' Label')
    df.columns = [c.strip() for c in df.columns]

    # Detect label column
    label_col = None
    for cand in ["ground_truth_label", "Label", "label", "attack_type"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col:
        df["ground_truth_label"] = df[label_col].apply(normalize_label)
        if label_col != "ground_truth_label":
            df.drop(columns=[label_col], inplace=True)
    elif drop_unlabeled:
        df["ground_truth_label"] = "Benign"

    # Deduplication
    if dedup_cols is None:
        # Default 5-tuple + timestamp + duration if present
        dedup_candidates = ["src_ip", "src_port", "dest_ip", "dest_port", "proto", "timestamp", "duration"]
        dedup_cols = [c for c in dedup_candidates if c in df.columns]

    if dedup_cols:
        initial_count = len(df)
        df.drop_duplicates(subset=dedup_cols, keep="first", inplace=True)

    # Clean duration if present
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0001)
        df["duration"] = df["duration"].apply(lambda d: max(float(d), 0.0001))

    # Replace inf and NaN across all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0.0)

    df.reset_index(drop=True, inplace=True)
    return df


def time_based_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically without random row shuffling to eliminate
    temporal leakage between train, validation, and test partitions.

    If multi-day data is present, assigns:
    - Monday–Wednesday: Train / Val
    - Thursday–Friday: Test
    Otherwise splits chronologically by timestamp quantile.

    Guarantees:
        max(train.timestamp) <= min(val.timestamp) <= min(test.timestamp)
    """
    df = df.copy()
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe.")

    # Convert timestamps to datetime for sorting (supporting ISO8601 with variable subseconds)
    df["_parsed_ts"] = pd.to_datetime(df[timestamp_col], format="ISO8601", utc=True, errors="coerce")
    # If any timestamps still failed to parse, fallback to mixed inference or index fill
    if df["_parsed_ts"].isna().any():
        df["_parsed_ts"] = pd.to_datetime(df[timestamp_col], format="mixed", utc=True, errors="coerce").ffill().bfill()

    # Sort strictly by chronological time
    df.sort_values(by="_parsed_ts", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Check day of week names
    days = df["_parsed_ts"].dt.day_name().unique()
    is_multi_day = len(days) > 1

    train_days = {"Monday", "Tuesday", "Wednesday"}
    test_days = {"Thursday", "Friday"}

    # If dataset spans multiple days matching CIC-IDS2017 week structure:
    if is_multi_day and (any(d in train_days for d in days) and any(d in test_days for d in days)):
        train_val_mask = df["_parsed_ts"].dt.day_name().isin(train_days)
        test_mask = df["_parsed_ts"].dt.day_name().isin(test_days)

        train_val_df = df[train_val_mask].copy()
        test_df = df[test_mask].copy()

        # Split train_val chronologically (80% train, 20% val)
        val_cutoff = int(len(train_val_df) * (1.0 - val_ratio / (train_ratio + val_ratio)))
        train_df = train_val_df.iloc[:val_cutoff].copy()
        val_df = train_val_df.iloc[val_cutoff:].copy()
    else:
        # Chronological quantile split
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

    # Tag partition column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df.drop(columns=["_parsed_ts"], inplace=True, errors="ignore")
    val_df.drop(columns=["_parsed_ts"], inplace=True, errors="ignore")
    test_df.drop(columns=["_parsed_ts"], inplace=True, errors="ignore")

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def prepare_ml_features_and_labels(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts strictly conforming feature matrix X and label vector y.
    If full feature columns are not yet present, computes them via extract_features_batch.
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES

    # Check if raw features need extraction from records
    missing_feats = [f for f in feature_cols if f not in df.columns]
    if missing_feats:
        # Extract features using shared feature extractor
        extracted = extract_features_batch(df.to_dict(orient="records"), include_metadata=False)
        X = extracted[feature_cols].copy()
    else:
        X = df[feature_cols].copy()

    # Ensure zero NaN or Inf
    for f in feature_cols:
        X[f] = pd.to_numeric(X[f], errors="coerce").fillna(0.0)
        X[f] = X[f].replace([np.inf, -np.inf], 0.0)

    y = df["ground_truth_label"] if "ground_truth_label" in df.columns else pd.Series(["Benign"] * len(df))
    return X, y
