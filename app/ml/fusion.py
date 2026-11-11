"""
Hybrid Fusion Engine for Intrusion Detection System
Fuses deterministic Suricata rule detections with ML behavioral anomaly inference.
Preserves deterministic explainability, high throughput, and zero packet drops.
"""

import json
import sqlite3
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from app.ml.features import (
    FEATURE_NAMES,
    extract_features_from_flow,
    extract_flow_metadata
)
from app.ml.model import HybridClassifier


class HybridFusionEngine:
    """
    Coordinates decision arbitration between Suricata signature alerts
    and ML flow anomaly classifications.
    """
    def __init__(
        self,
        classifier: Optional[HybridClassifier] = None,
        model_path: str = "models/hybrid_model.joblib"
    ):
        if classifier is not None:
            self.classifier = classifier
        elif os.path.exists(model_path):
            self.classifier = HybridClassifier.load(model_path)
        else:
            self.classifier = HybridClassifier(use_rf=True)

        self.alert_index = defaultdict(list)

    def register_alerts(self, alerts: List[Dict[str, Any]]):
        """Indexes active Suricata alerts for fast O(1) flow correlation."""
        self.alert_index.clear()
        for a in alerts:
            s_ip = str(a.get("src_ip", ""))
            s_port = int(a.get("src_port", 0))
            d_ip = str(a.get("dest_ip", ""))
            d_port = int(a.get("dest_port", 0))
            proto = str(a.get("proto", "TCP")).upper()
            key = (s_ip, s_port, d_ip, d_port, proto)
            self.alert_index[key].append(a)

    def find_matching_alert(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Finds any Suricata signature alert matching the given flow 5-tuple."""
        s_ip = metadata.get("src_ip", "")
        s_port = metadata.get("src_port", 0)
        d_ip = metadata.get("dest_ip", "")
        d_port = metadata.get("dest_port", 0)
        proto = metadata.get("proto", "TCP").upper()

        key = (s_ip, s_port, d_ip, d_port, proto)
        candidates = self.alert_index.get(key, [])
        if candidates:
            # Return alert with highest severity (lowest numeric value)
            return sorted(candidates, key=lambda x: int(x.get("severity", 3)))[0]
        return None

    def process_flow(
        self,
        event: Union[Dict[str, Any], str],
        matching_alert: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes hybrid fusion logic on a single Suricata EVE flow event.

        Resolves one of 4 states:
        - CORROBORATED: Signature matched + ML predicted attack (dual explanation).
        - SIGNATURE_ONLY: Signature matched + ML baseline (rule takes precedence).
        - ANOMALY_ONLY: Signature silent + ML anomaly (evasion/zero-day alert).
        - BENIGN: Signature silent + ML baseline (verified clean).
        """
        meta = extract_flow_metadata(event)
        features = extract_features_from_flow(event)

        # ML inference
        ml_pred = self.classifier.predict_single(features)
        pred_label = ml_pred["predicted_label"]
        is_anomaly = ml_pred["is_anomaly"]
        confidence = ml_pred["confidence"]
        attack_prob = ml_pred["attack_probability"]
        top_features = ml_pred["top_contributing_features"]

        # Alert correlation
        if matching_alert is None:
            matching_alert = self.find_matching_alert(meta)

        has_alert = matching_alert is not None
        sig_id = int(matching_alert.get("signature_id", 0)) if has_alert else 0
        sig_name = str(matching_alert.get("signature", "")) if has_alert else ""

        # Construct feature attribution string for explainability
        feat_str_list = [
            f"{f['feature']}={f['observed_value']} (z={f['z_score']})"
            for f in top_features
        ]
        feat_str = ", ".join(feat_str_list) if feat_str_list else "nominal"

        # Hybrid Decision Logic
        if has_alert and is_anomaly == 1:
            fusion_status = "CORROBORATED"
            explanation = (
                f"[CORROBORATED ATTACK] Signature SID {sig_id} ('{sig_name}') confirmed by Suricata. "
                f"ML model independently corroborates attack pattern ({pred_label}, {attack_prob*100:.1f}% confidence). "
                f"Key anomalous features: {feat_str}."
            )
        elif has_alert and is_anomaly == 0:
            fusion_status = "SIGNATURE_ONLY"
            explanation = (
                f"[SIGNATURE DETECTED] Signature SID {sig_id} ('{sig_name}') triggered. "
                f"ML assessed flow behavior as baseline/benign; deterministic rule criteria met and takes precedence."
            )
        elif not has_alert and is_anomaly == 1:
            fusion_status = "ANOMALY_ONLY"
            explanation = (
                f"[BEHAVIORAL ANOMALY] Suricata rules silent (suspected zero-day or evasion). "
                f"ML classifier flagged statistical deviation ({pred_label}, {attack_prob*100:.1f}% confidence). "
                f"Top anomalous features: {feat_str}."
            )
        else:
            fusion_status = "BENIGN"
            explanation = "[BENIGN] Flow verified clean by both Suricata signature engine and ML baseline."

        return {
            "flow_id": meta["flow_id"],
            "timestamp": meta["timestamp"],
            "src_ip": meta["src_ip"],
            "src_port": meta["src_port"],
            "dest_ip": meta["dest_ip"],
            "dest_port": meta["dest_port"],
            "proto": meta["proto"],
            "predicted_label": pred_label,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "attack_probability": attack_prob,
            "top_contributing_features": json.dumps(top_features),
            "fusion_status": fusion_status,
            "signature_id": sig_id,
            "signature": sig_name,
            "explainability_summary": explanation
        }

    def batch_process_flows(
        self,
        flows: List[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Processes a batch of flows with indexed alert lookup."""
        if alerts is not None:
            self.register_alerts(alerts)

        results = []
        for flow in flows:
            record = self.process_flow(flow)
            results.append(record)
        return results

    @staticmethod
    def persist_fusion_records(
        records: List[Dict[str, Any]],
        db_path: str = "database/alerts.db"
    ) -> int:
        """Persists fusion records to SQLite using high-throughput WAL mode."""
        if not records:
            return 0

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")

        insert_sql = """
            INSERT INTO ml_predictions (
                flow_id, timestamp, src_ip, src_port, dest_ip, dest_port, proto,
                predicted_label, is_anomaly, confidence, attack_probability,
                top_contributing_features, fusion_status, signature_id, signature,
                explainability_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        tuples = [
            (
                r["flow_id"], r["timestamp"], r["src_ip"], r["src_port"],
                r["dest_ip"], r["dest_port"], r["proto"], r["predicted_label"],
                r["is_anomaly"], r["confidence"], r["attack_probability"],
                r["top_contributing_features"], r["fusion_status"],
                r["signature_id"], r["signature"], r["explainability_summary"]
            )
            for r in records
        ]

        cursor.executemany(insert_sql, tuples)
        conn.commit()
        conn.close()
        return len(tuples)

