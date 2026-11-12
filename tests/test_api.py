"""
Unit tests for FastAPI endpoints (app/server.py)
Tests ML metrics, hybrid fusion summaries, predictions queries, and database reset.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from app.server import app, DB_PATH
from src.eve_ingestor import init_database
from app.ml.fusion import HybridFusionEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_get_ml_metrics_endpoint(client):
    response = client.get("/api/ml/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_summary" in data or "status" in data
    if "evaluation_summary" in data:
        summary = data["evaluation_summary"]
        assert "macro_f1" in summary
        assert "macro_pr_auc" in summary
        assert "accuracy" in summary
        assert "benign_false_alarm_rate" in summary
        assert "per_class_metrics" in data
        assert "confusion_matrix" in data


def test_get_fusion_summary_endpoint(client):
    response = client.get("/api/ml/fusion/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "corroborated" in data
    assert "anomaly_only" in data
    assert "signature_only" in data
    assert "benign" in data


def test_get_ml_predictions_endpoint(client):
    response = client.get("/api/ml/predictions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total" in data
    assert isinstance(data["predictions"], list)


def test_reset_endpoint(client):
    response = client.post("/api/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_benchmark_endpoints(client):
    # Test GET benchmark results
    res_get = client.get("/api/benchmark/results?split=all")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["status"] == "verified"
    assert "exact_accuracy" in data_get
    assert "macro_f1" in data_get
    assert "confusion_matrix" in data_get
    assert "category_metrics" in data_get
    assert "mae" in data_get
    assert "rmse" in data_get
    assert data_get["total_flows"] == 6216

    # Test POST benchmark run with held-out test split
    res_run = client.post("/api/benchmark/run?split=test")
    assert res_run.status_code == 200
    data_run = res_run.json()
    assert data_run["status"] == "verified"
    assert data_run["total_flows"] == 933
    assert data_run["exact_accuracy"] > 0.0
    assert len(data_run["confusion_matrix"]["classes"]) > 0


