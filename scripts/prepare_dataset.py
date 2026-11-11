"""
CLI script to clean dataset, extract ML features, and perform time-based splitting.
Generates data/splits/train.csv, data/splits/val.csv, data/splits/test.csv.
"""

import os
import sys
import argparse
import pandas as pd

# Ensure repo root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ml.dataset import clean_dataset, time_based_split, normalize_label
from app.ml.features import extract_features_batch, FEATURE_NAMES, METADATA_NAMES


def main():
    parser = argparse.ArgumentParser(description="Prepare, clean, and split ML dataset")
    parser.add_argument("--input", "-i", type=str, default="data/ground_truth.csv", help="Input ground truth CSV")
    parser.add_argument("--output-dir", "-o", type=str, default="data/splits", help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"[*] Loading dataset from: {args.input}")
    raw_df = pd.read_csv(args.input)
    print(f"[*] Raw dataset shape: {raw_df.shape}")

    # 1. Clean dataset
    print("[*] Cleaning data: dropping duplicates, handling inf/NaN, normalizing labels...")
    cleaned_df = clean_dataset(raw_df)
    print(f"[+] Cleaned dataset shape: {cleaned_df.shape}")
    print("[*] Label distribution:")
    print(cleaned_df["ground_truth_label"].value_counts())

    # 2. Extract features using the shared feature extraction engine
    print(f"[*] Extracting full {len(FEATURE_NAMES)} features using shared engine...")
    records = cleaned_df.to_dict(orient="records")
    features_df = extract_features_batch(records, include_metadata=True)
    features_df["ground_truth_label"] = cleaned_df["ground_truth_label"].values

    # 3. Time-based split
    print(f"[*] Performing time-based split (Train: {args.train_ratio*100:.0f}%, Val: {args.val_ratio*100:.0f}%, Test: {args.test_ratio*100:.0f}%)...")
    train_df, val_df, test_df = time_based_split(
        features_df,
        timestamp_col="timestamp",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    print(f"[+] Train split: {len(train_df)} rows ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"[+] Val split:   {len(val_df)} rows ({val_df['timestamp'].min()} -> {val_df['timestamp'].max()})")
    print(f"[+] Test split:  {len(test_df)} rows ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")

    # 4. Save splits
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Splits saved successfully to '{args.output_dir}':")
    print(f"    - {train_path} ({len(train_df)} rows)")
    print(f"    - {val_path} ({len(val_df)} rows)")
    print(f"    - {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()

