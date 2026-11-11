"""
Scientific Evaluation & Metric Reporting Engine for Machine Learning IDS Layer
Computes per-class Precision, Recall, F1, PR-AUC, Confusion Matrix, and False Alarm Rates.
Adheres strictly to the requirement: NEVER report accuracy alone.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)

from app.ml.features import FEATURE_NAMES


def compute_multiclass_pr_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray
) -> Dict[str, float]:
    """
    Computes Precision-Recall AUC (PR-AUC) for each class using One-vs-Rest.
    Returns per-class PR-AUC and macro average.
    """
    pr_aucs: Dict[str, float] = {}
    macro_list = []

    for i, cls_name in enumerate(classes):
        # Binary target for class i
        y_binary = (y_true == cls_name).astype(int)
        # Probabilities for class i
        probs = y_proba[:, i] if y_proba.shape[1] > i else np.zeros_like(y_binary, dtype=float)

        if np.sum(y_binary) == 0:
            pr_aucs[cls_name] = 0.0
            continue

        precision, recall, _ = precision_recall_curve(y_binary, probs)
        pr_score = float(auc(recall, precision))
        pr_aucs[cls_name] = round(pr_score, 4)
        macro_list.append(pr_score)

    pr_aucs["macro_avg"] = round(float(np.mean(macro_list)), 4) if macro_list else 0.0
    return pr_aucs


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    classes: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Comprehensive scientific evaluation of ML model on held-out test data.

    Returns:
        Dict containing:
        - per_class_metrics (Precision, Recall, F1, PR-AUC, Support)
        - macro_metrics (Macro Precision, Recall, F1, PR-AUC)
        - confusion_matrix (classes, matrix, breakdown)
        - benign_false_alarm_rate
        - inference_latency_us
        - exact_accuracy (explicitly reported in context alongside F1/PR-AUC)
    """
    if isinstance(X_test, pd.DataFrame):
        X_mat = X_test[FEATURE_NAMES].values
    else:
        X_mat = np.asarray(X_test)

    y_arr = np.asarray(y_test)
    if classes is None:
        classes = model.classes_

    # Benchmark single inference latency
    t0 = time.perf_counter()
    y_pred = model.predict(X_mat)
    y_proba = model.predict_proba(X_mat)
    elapsed_total = time.perf_counter() - t0
    latency_us = round((elapsed_total / max(len(X_mat), 1)) * 1_000_000, 2)

    # 1. Per-Class Precision, Recall, F1
    report_dict = classification_report(
        y_arr,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    # 2. Per-Class PR-AUC
    pr_auc_dict = compute_multiclass_pr_auc(y_arr, y_proba, classes)

    per_class_summary = []
    for cls_name in classes:
        cls_rep = report_dict.get(cls_name, {})
        per_class_summary.append({
            "class_name": cls_name,
            "precision": round(float(cls_rep.get("precision", 0.0)), 4),
            "recall": round(float(cls_rep.get("recall", 0.0)), 4),
            "f1_score": round(float(cls_rep.get("f1-score", 0.0)), 4),
            "pr_auc": pr_auc_dict.get(cls_name, 0.0),
            "support": int(cls_rep.get("support", 0))
        })

    # 3. Macro & Weighted Metrics
    macro_precision = round(float(report_dict.get("macro avg", {}).get("precision", 0.0)), 4)
    macro_recall = round(float(report_dict.get("macro avg", {}).get("recall", 0.0)), 4)
    macro_f1 = round(float(report_dict.get("macro avg", {}).get("f1-score", 0.0)), 4)
    macro_pr_auc = pr_auc_dict.get("macro_avg", 0.0)

    # 4. Multi-Class Confusion Matrix
    cm_array = confusion_matrix(y_arr, y_pred, labels=classes)
    cm_dict = {
        "classes": list(classes),
        "matrix": cm_array.tolist(),
        "total_test_samples": int(len(y_arr))
    }

    # 5. Benign Specificity and False Alarm Rate (FAR)
    benign_far = 0.0
    if "Benign" in classes:
        b_idx = list(classes).index("Benign")
        benign_total = np.sum(y_arr == "Benign")
        benign_fp = np.sum((y_arr == "Benign") & (y_pred != "Benign"))
        benign_far = round(float(benign_fp / max(benign_total, 1)), 4)

    # Exact accuracy (never alone!)
    accuracy = round(float(np.mean(y_pred == y_arr)), 4)

    return {
        "evaluation_summary": {
            "total_samples": len(y_arr),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_pr_auc": macro_pr_auc,
            "benign_false_alarm_rate": benign_far,
            "accuracy": accuracy,
            "inference_latency_us_per_flow": latency_us,
            "note": "Evaluated strictly with per-class Precision, Recall, F1, PR-AUC, and Confusion Matrix to prevent class-imbalance bias."
        },
        "per_class_metrics": per_class_summary,
        "confusion_matrix": cm_dict
    }


def format_evaluation_table(eval_results: Dict[str, Any]) -> str:
    """Returns a cleanly formatted ASCII table of the evaluation results."""
    summary = eval_results["evaluation_summary"]
    rows = eval_results["per_class_metrics"]
    cm = eval_results["confusion_matrix"]

    lines = [
        "=" * 78,
        "  SCIENTIFIC ML EVALUATION REPORT (HELD-OUT TEST SET)",
        "=" * 78,
        f"Total Samples: {summary['total_samples']:,} | Single Flow Latency: {summary['inference_latency_us_per_flow']} µs",
        f"Macro F1-Score: {summary['macro_f1']*100:.2f}% | Macro PR-AUC: {summary['macro_pr_auc']*100:.2f}%",
        f"Macro Precision: {summary['macro_precision']*100:.2f}% | Macro Recall: {summary['macro_recall']*100:.2f}%",
        f"Benign False Alarm Rate (FAR): {summary['benign_false_alarm_rate']*100:.2f}% | Accuracy: {summary['accuracy']*100:.2f}%",
        "-" * 78,
        f"{'Class Name':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'PR-AUC':<10} {'Support':<8}",
        "-" * 78
    ]

    for r in rows:
        lines.append(
            f"{r['class_name']:<30} {r['precision']:<10.4f} {r['recall']:<10.4f} "
            f"{r['f1_score']:<10.4f} {r['pr_auc']:<10.4f} {r['support']:<8}"
        )

    lines.extend([
        "-" * 78,
        "Confusion Matrix:",
        f"Classes: {cm['classes']}",
        str(np.array(cm['matrix'])),
        "=" * 78
    ])
    return "\n".join(lines)

