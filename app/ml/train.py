"""
Model Training Pipeline for Hybrid IDS
Trains the LightGBM / Random Forest hybrid classifier on chronologically split data,
evaluates on held-out test data, and persists artifacts for real-time inference.
"""

import os
import sys
import json
import argparse
import pandas as pd

# Ensure repo root on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ml.features import FEATURE_NAMES
from app.ml.dataset import clean_dataset, time_based_split
from app.ml.model import HybridClassifier
from app.ml.evaluate import evaluate_model, format_evaluation_table


def train_pipeline(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    model_output_path: str = "models/hybrid_model.joblib",
    report_output_path: str = "models/evaluation_report.json",
    use_rf: bool = False
) -> HybridClassifier:
    """End-to-end training and evaluation pipeline."""
    # Ensure splits exist
    for path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(path):
            print(f"[*] Missing '{path}'. Generating splits from data/ground_truth.csv...")
            from scripts.prepare_dataset import main as run_prep
            # Invoke dataset preparation
            run_prep()
            break

    print(f"[*] Loading training data from: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"[*] Loading validation data from: {val_csv}")
    val_df = pd.read_csv(val_csv)
    print(f"[*] Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["ground_truth_label"]

    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["ground_truth_label"]

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["ground_truth_label"]

    print(f"[*] Training dataset shape: {X_train.shape} across classes: {train_df['ground_truth_label'].unique().tolist()}")
    print("[*] Initializing HybridClassifier (LightGBM with feature attribution)...")
    classifier = HybridClassifier(use_rf=use_rf)

    print("[*] Fitting model on training split...")
    classifier.fit(X_train, y_train)

    print("[*] Evaluating on held-out validation split...")
    val_eval = evaluate_model(classifier, X_val, y_val)
    print(f"[+] Validation Macro F1: {val_eval['evaluation_summary']['macro_f1']*100:.2f}% | PR-AUC: {val_eval['evaluation_summary']['macro_pr_auc']*100:.2f}%")

    print("[*] Evaluating on strictly held-out chronological test split...")
    test_eval = evaluate_model(classifier, X_test, y_test)
    report_str = format_evaluation_table(test_eval)
    print("\n" + report_str)

    # Persist model and metrics
    print(f"[*] Persisting model artifact to: {model_output_path}")
    classifier.save(model_output_path)

    os.makedirs(os.path.dirname(os.path.abspath(report_output_path)), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(test_eval, f, indent=2)
    print(f"[+] Scientific evaluation report saved to: {report_output_path}")

    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid IDS ML Model")
    parser.add_argument("--train-csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--model-out", type=str, default="models/hybrid_model.joblib")
    parser.add_argument("--report-out", type=str, default="models/evaluation_report.json")
    parser.add_argument("--use-rf", action="store_true", help="Force Random Forest instead of LightGBM")
    args = parser.parse_args()

    train_pipeline(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        model_output_path=args.model_out,
        report_output_path=args.report_out,
        use_rf=args.use_rf
    )
