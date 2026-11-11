"""
Unit tests for app.ml.fusion
Tests hybrid decision arbitration (CORROBORATED, SIGNATURE_ONLY, ANOMALY_ONLY, BENIGN),
deterministic dual-explainability strings, alert indexing, and SQLite WAL persistence.
"""

import os
import json
import sqlite3
import tempfile
import pytest
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_NAMES
from app.ml.model import HybridClassifier
from app.ml.fusion import HybridFusionEngine


class MockClassifier:
    """Mock classifier allowing deterministic control over predict_single outputs."""
    def __init__(self, pred_label="Benign", is_anomaly=0, conf=0.95, attack_prob=0.05):
        self.pred_label = pred_label
        self.is_anomaly = is_anomaly
        self.conf = conf
        self.attack_prob = attack_prob
        self.classes_ = np.array(["Benign", "DoS-LOIC-UDP"])

    def predict_single(self, features):
        return {
            "predicted_label": self.pred_label,
            "is_anomaly": self.is_anomaly,
            "confidence": self.conf,
            "attack_probability": self.attack_prob,
            "top_contributing_features": [
                {"feature": "pkts_per_sec", "observed_value": 450.0, "baseline_mean": 10.0, "z_score": 4.5},
                {"feature": "bytes_per_sec", "observed_value": 50000.0, "baseline_mean": 2000.0, "z_score": 3.2}
            ],
            "features_snapshot": {}
        }


@pytest.fixture
def mock_flow():
    return {
        "flow_id": "FLOW_TEST_001",
        "timestamp": "2017-07-06T13:20:00Z",
        "src_ip": "172.16.0.1",
        "src_port": 49152,
        "dest_ip": "192.168.10.50",
        "dest_port": 80,
        "proto": "TCP",
        "flow": {
            "pkts_toserver": 100,
            "bytes_toserver": 5000,
            "age": 1.0
        }
    }


def test_fusion_corroborated(mock_flow):
    classifier = MockClassifier(pred_label="DoS-LOIC-UDP", is_anomaly=1, conf=0.98, attack_prob=0.98)
    engine = HybridFusionEngine(classifier=classifier)

    alert = {
        "signature_id": 1000002,
        "signature": "DoS High Volume TCP Syn Flood",
        "severity": 1,
        "src_ip": "172.16.0.1",
        "src_port": 49152,
        "dest_ip": "192.168.10.50",
        "dest_port": 80,
        "proto": "TCP"
    }

    res = engine.process_flow(mock_flow, matching_alert=alert)
    assert res["fusion_status"] == "CORROBORATED"
    assert res["signature_id"] == 1000002
    assert "CORROBORATED ATTACK" in res["explainability_summary"]
    assert "DoS-LOIC-UDP" in res["explainability_summary"]
    assert "pkts_per_sec" in res["explainability_summary"]


def test_fusion_signature_only(mock_flow):
    classifier = MockClassifier(pred_label="Benign", is_anomaly=0, conf=0.92, attack_prob=0.08)
    engine = HybridFusionEngine(classifier=classifier)

    alert = {
        "signature_id": 1000006,
        "signature": "Port Scan TCP SYN Probing",
        "severity": 2,
        "src_ip": "172.16.0.1",
        "src_port": 49152,
        "dest_ip": "192.168.10.50",
        "dest_port": 80,
        "proto": "TCP"
    }

    res = engine.process_flow(mock_flow, matching_alert=alert)
    assert res["fusion_status"] == "SIGNATURE_ONLY"
    assert res["signature_id"] == 1000006
    assert "SIGNATURE DETECTED" in res["explainability_summary"]
    assert "takes precedence" in res["explainability_summary"]


def test_fusion_anomaly_only(mock_flow):
    classifier = MockClassifier(pred_label="Web Attack - SQL Injection", is_anomaly=1, conf=0.88, attack_prob=0.88)
    engine = HybridFusionEngine(classifier=classifier)

    # No alert provided
    res = engine.process_flow(mock_flow, matching_alert=None)
    assert res["fusion_status"] == "ANOMALY_ONLY"
    assert res["signature_id"] == 0
    assert "BEHAVIORAL ANOMALY" in res["explainability_summary"]
    assert "zero-day or evasion" in res["explainability_summary"]


def test_fusion_benign(mock_flow):
    classifier = MockClassifier(pred_label="Benign", is_anomaly=0, conf=0.99, attack_prob=0.01)
    engine = HybridFusionEngine(classifier=classifier)

    res = engine.process_flow(mock_flow, matching_alert=None)
    assert res["fusion_status"] == "BENIGN"
    assert "clean" in res["explainability_summary"].lower()


def test_persist_fusion_records_and_wal():
    classifier = MockClassifier()
    engine = HybridFusionEngine(classifier=classifier)

    records = [
        {
            "flow_id": "F1", "timestamp": "2017-07-06T12:00:00Z",
            "src_ip": "10.0.0.1", "src_port": 1234, "dest_ip": "10.0.0.2", "dest_port": 80, "proto": "TCP",
            "predicted_label": "Benign", "is_anomaly": 0, "confidence": 0.99, "attack_probability": 0.01,
            "top_contributing_features": "[]", "fusion_status": "BENIGN",
            "signature_id": 0, "signature": "", "explainability_summary": "[BENIGN] Clean"
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_alerts.db")
        # Initialize schema
        from src.eve_ingestor import init_database
        init_database(db_path=db_path)

        inserted = HybridFusionEngine.persist_fusion_records(records, db_path=db_path)
        assert inserted == 1

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        mode = cur.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

        row = cur.execute("SELECT flow_id, fusion_status, predicted_label FROM ml_predictions WHERE flow_id = 'F1'").fetchone()
        assert row is not None
        assert row[0] == "F1"
        assert row[1] == "BENIGN"
        conn.close()

