"""
Unit tests for app.ml.dataset
Tests data cleaning, label standardization, deduplication, Inf/NaN handling,
and strict temporal day/time-based splits.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import (
    clean_dataset,
    normalize_label,
    time_based_split,
    prepare_ml_features_and_labels,
    LABEL_NORMALIZATION_MAP
)
from app.ml.features import FEATURE_NAMES


def test_normalize_label():
    assert normalize_label("benign") == "Benign"
    assert normalize_label("normal") == "Benign"
    assert normalize_label(None) == "Benign"
    assert normalize_label(np.nan) == "Benign"
    assert normalize_label("  dos-loic-udp  ") == "DoS-LOIC-UDP"
    assert normalize_label("Web Attack - SQL Injection") == "Web Attack - SQL Injection"
    assert normalize_label("webattack-xss") == "Web Attack - XSS"
    assert normalize_label("botnet") == "Botnet-C2-Beaconing"
    assert normalize_label("heartbleed") == "DoS-Slowloris"


def test_clean_dataset_deduplication():
    data = [
        {"src_ip": "10.0.0.1", "src_port": 50000, "dest_ip": "10.0.0.2", "dest_port": 80, "proto": "TCP", "timestamp": "2017-07-06T10:00:00Z", "duration": 1.0, "ground_truth_label": "Benign"},
        {"src_ip": "10.0.0.1", "src_port": 50000, "dest_ip": "10.0.0.2", "dest_port": 80, "proto": "TCP", "timestamp": "2017-07-06T10:00:00Z", "duration": 1.0, "ground_truth_label": "Benign"}, # Duplicate
        {"src_ip": "10.0.0.1", "src_port": 50001, "dest_ip": "10.0.0.2", "dest_port": 80, "proto": "TCP", "timestamp": "2017-07-06T10:00:01Z", "duration": 1.0, "ground_truth_label": "Benign"}
    ]
    df = pd.DataFrame(data)
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2


def test_clean_dataset_inf_nan_and_duration():
    data = [
        {"timestamp": "2017-07-06T10:00:00Z", "duration": -5.0, "bytes_per_sec": np.inf, "pkts_per_sec": -np.inf, "val": np.nan, "Label": "benign"},
        {"timestamp": "2017-07-06T10:00:01Z", "duration": 0.0, "bytes_per_sec": 100.0, "pkts_per_sec": 20.0, "val": 1.0, "Label": "dos-slowloris"}
    ]
    df = pd.DataFrame(data)
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2
    assert cleaned.loc[0, "duration"] == 0.0001
    assert cleaned.loc[1, "duration"] == 0.0001
    assert not np.isinf(cleaned["bytes_per_sec"]).any()
    assert not np.isinf(cleaned["pkts_per_sec"]).any()
    assert not cleaned.isna().any().any()
    assert cleaned.loc[0, "ground_truth_label"] == "Benign"
    assert cleaned.loc[1, "ground_truth_label"] == "DoS-Slowloris"


def test_time_based_split_chronological_ordering():
    # 100 sequential timestamps across 1 day
    timestamps = pd.date_range("2017-07-06 09:00:00", periods=100, freq="min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in timestamps],
        "duration": [1.0] * 100,
        "dest_port": [80] * 100,
        "proto": ["TCP"] * 100,
        "ground_truth_label": ["Benign"] * 50 + ["Web Attack - Brute Force"] * 50
    })

    # Shuffle the input dataframe to test that time_based_split restores strict chronological order
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_df, val_df, test_df = time_based_split(
        df_shuffled,
        timestamp_col="timestamp",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Verify temporal guarantee: max(train) <= min(val) <= max(val) <= min(test)
    train_max = pd.to_datetime(train_df["timestamp"], utc=True).max()
    val_min = pd.to_datetime(val_df["timestamp"], utc=True).min()
    val_max = pd.to_datetime(val_df["timestamp"], utc=True).max()
    test_min = pd.to_datetime(test_df["timestamp"], utc=True).min()

    assert train_max <= val_min, "Leakage detected: train_max > val_min"
    assert val_max <= test_min, "Leakage detected: val_max > test_min"
    assert set(train_df["split"].unique()) == {"train"}
    assert set(val_df["split"].unique()) == {"val"}
    assert set(test_df["split"].unique()) == {"test"}


def test_time_based_split_multi_day_assignment():
    # 5 days: Monday through Friday (CIC-IDS2017 schedule)
    dates = [
        "2017-07-03T10:00:00Z",  # Monday (Benign)
        "2017-07-04T10:00:00Z",  # Tuesday (Brute Force)
        "2017-07-05T10:00:00Z",  # Wednesday (DoS)
        "2017-07-06T10:00:00Z",  # Thursday (Web Attack)
        "2017-07-07T10:00:00Z"   # Friday (Botnet/DDoS)
    ]
    df = pd.DataFrame({
        "timestamp": dates,
        "dest_port": [80, 22, 80, 80, 80],
        "ground_truth_label": ["Benign", "BruteForce-SSH-Patator", "DoS-LOIC-UDP", "Web Attack - SQL Injection", "Botnet-C2-Beaconing"]
    })

    train_df, val_df, test_df = time_based_split(df, timestamp_col="timestamp")

    # Monday-Wednesday must be in train/val
    train_val_timestamps = pd.to_datetime(pd.concat([train_df["timestamp"], val_df["timestamp"]]), utc=True)
    train_val_days = train_val_timestamps.dt.day_name().unique()
    for d in train_val_days:
        assert d in ["Monday", "Tuesday", "Wednesday"]

    # Thursday-Friday must be in test
    test_timestamps = pd.to_datetime(test_df["timestamp"], utc=True)
    test_days = test_timestamps.dt.day_name().unique()
    for d in test_days:
        assert d in ["Thursday", "Friday"]


def test_prepare_ml_features_and_labels():
    records = [
        {"dest_port": 80, "proto": "TCP", "flow": {"pkts_toserver": 5, "bytes_toserver": 500, "age": 1.0}, "ground_truth_label": "Benign"},
        {"dest_port": 22, "proto": "TCP", "flow": {"pkts_toserver": 20, "bytes_toserver": 2000, "age": 2.0}, "ground_truth_label": "BruteForce-SSH-Patator"}
    ]
    df = pd.DataFrame(records)
    X, y = prepare_ml_features_and_labels(df)

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == 2
    assert len(y) == 2
    assert y.tolist() == ["Benign", "BruteForce-SSH-Patator"]
    assert not X.isna().any().any()
    assert not np.isinf(X.to_numpy()).any()
