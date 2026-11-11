"""
Unit tests for app.ml.model and app.ml.evaluate
Tests model training, single-record inference, explainability attribution,
serialization roundtrips, and scientific evaluation metric computation.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from app.ml.features import FEATURE_NAMES
from app.ml.model import HybridClassifier
from app.ml.evaluate import evaluate_model, compute_multiclass_pr_auc


@pytest.fixture
def sample_dataset():
    """Generates synthetic dataset for training and testing model operations."""
    np.random.seed(42)
    n = 100
    data = {f: np.random.uniform(0.0, 10.0, size=n) for f in FEATURE_NAMES}
    # Make duration > 0
    data["duration"] = np.random.uniform(0.1, 5.0, size=n)
    # Distinguish classes with specific features
    data["pkts_per_sec"][:50] = np.random.uniform(1.0, 10.0, size=50)   # Benign
    data["pkts_per_sec"][50:] = np.random.uniform(50.0, 100.0, size=50) # Attack

    df = pd.DataFrame(data)
    y = np.array(["Benign"] * 50 + ["DoS-LOIC-UDP"] * 50)
    return df, y


def test_model_fit_and_predict_batch(sample_dataset):
    X, y = sample_dataset
    classifier = HybridClassifier(use_rf=True, random_state=42)
    classifier.fit(X, y)

    preds = classifier.predict(X)
    probas = classifier.predict_proba(X)

    assert len(preds) == len(X)
    assert probas.shape == (len(X), 2)
    assert set(classifier.classes_) == {"Benign", "DoS-LOIC-UDP"}


def test_predict_single_and_explainability(sample_dataset):
    X, y = sample_dataset
    classifier = HybridClassifier(use_rf=True, random_state=42)
    classifier.fit(X, y)

    flow_record = {
        "dest_port": 80,
        "proto": "TCP",
        "flow": {
            "pkts_toserver": 1000,
            "bytes_toserver": 50000,
            "age": 0.5
        }
    }

    result = classifier.predict_single(flow_record)
    assert "predicted_label" in result
    assert "is_anomaly" in result
    assert "confidence" in result
    assert "attack_probability" in result
    assert "top_contributing_features" in result

    explanations = result["top_contributing_features"]
    assert len(explanations) <= 3
    for exp in explanations:
        assert "feature" in exp
        assert "observed_value" in exp
        assert "baseline_mean" in exp
        assert "z_score" in exp


def test_model_save_and_load_roundtrip(sample_dataset):
    X, y = sample_dataset
    classifier = HybridClassifier(use_rf=True, random_state=42)
    classifier.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.joblib")
        classifier.save(model_path)
        assert os.path.exists(model_path)

        loaded = HybridClassifier.load(model_path)
        preds_orig = classifier.predict(X[:5])
        preds_loaded = loaded.predict(X[:5])
        np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_scientific_evaluation_metrics(sample_dataset):
    X, y = sample_dataset
    classifier = HybridClassifier(use_rf=True, random_state=42)
    classifier.fit(X, y)

    eval_results = evaluate_model(classifier, X, y)

    # Verify requirement: never report accuracy alone
    summary = eval_results["evaluation_summary"]
    assert "macro_precision" in summary
    assert "macro_recall" in summary
    assert "macro_f1" in summary
    assert "macro_pr_auc" in summary
    assert "benign_false_alarm_rate" in summary
    assert "inference_latency_us_per_flow" in summary

    # Per class metrics check
    per_class = eval_results["per_class_metrics"]
    assert len(per_class) == 2
    for c in per_class:
        assert "precision" in c
        assert "recall" in c
        assert "f1_score" in c
        assert "pr_auc" in c
        assert 0.0 <= c["pr_auc"] <= 1.0

    # Confusion matrix check
    cm = eval_results["confusion_matrix"]
    assert "classes" in cm
    assert "matrix" in cm
    assert len(cm["matrix"]) == 2


def test_compute_multiclass_pr_auc():
    y_true = np.array(["Benign", "Attack", "Benign", "Attack"])
    classes = np.array(["Benign", "Attack"])
    # Perfect probabilities
    y_proba = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.85, 0.15],
        [0.05, 0.95]
    ])
    pr_aucs = compute_multiclass_pr_auc(y_true, y_proba, classes)
    assert "Benign" in pr_aucs
    assert "Attack" in pr_aucs
    assert "macro_avg" in pr_aucs
    assert pr_aucs["Benign"] > 0.8
    assert pr_aucs["Attack"] > 0.8
