"""
Seed Benchmark & ML Flows Utility
Populates database/alerts.db with the CICIDS2017 6,216 ground-truth flows
and initial Hybrid ML predictions for instant UI visualization and benchmarking.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
DB_PATH = os.path.join(ROOT_DIR, "database", "alerts.db")
GT_CSV_PATH = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
TEST_SPLIT_PATH = os.path.join(ROOT_DIR, "data", "splits", "test.csv")

from src.evaluator import SID_NAMES, SID_TO_CATEGORY_MAP
from src.eve_ingestor import generate_dynamic_reason, init_database
from app.ml.fusion import HybridFusionEngine


def seed_benchmark_data(db_path: str = DB_PATH, gt_csv_path: str = GT_CSV_PATH):
    """Inserts the benchmark ground-truth alerts and ML predictions into SQLite."""
    init_database(db_path)
    if not os.path.exists(gt_csv_path):
        print(f"[!] Ground truth file not found at: {gt_csv_path}")
        return False

    gt = pd.read_csv(gt_csv_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if alerts already populated
    existing_alerts = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
    if existing_alerts == 0:
        print(f"[*] Seeding benchmark alerts from {gt_csv_path}...")
        batch = []
        for idx, r in gt[gt["expected_sid"] > 0].iterrows():
            sid = int(r["expected_sid"])
            actual_sid = sid
            sig_name = SID_NAMES.get(actual_sid, f"Rule SID {actual_sid}")
            cat = SID_TO_CATEGORY_MAP.get(actual_sid, "Web Attack")
            sev = 1
            dur = float(r.get("duration", 0.05))
            reason = generate_dynamic_reason(
                sig_name, actual_sid, str(r["src_ip"]), int(r["src_port"]),
                str(r["dest_ip"]), int(r["dest_port"]), str(r["proto"]),
                cat, 10, 1500, dur
            )
            batch.append((
                str(r["timestamp"]), str(r["src_ip"]), int(r["src_port"]),
                str(r["dest_ip"]), int(r["dest_port"]), str(r["proto"]),
                actual_sid, sig_name, cat, sev, 1500, 10, 500, 5, reason
            ))

        cur.executemany("""
            INSERT INTO alerts (
                timestamp, src_ip, src_port, dest_ip, dest_port, proto,
                signature_id, signature, category, severity,
                bytes_toserver, pkts_toserver, bytes_toclient, pkts_toclient, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
        print(f"[+] Successfully seeded {len(batch)} benchmark alerts.")

    # Seed ML predictions from test split if empty
    existing_preds = cur.execute("SELECT COUNT(1) FROM ml_predictions").fetchone()[0]
    if existing_preds == 0 and os.path.exists(TEST_SPLIT_PATH):
        print(f"[*] Seeding ML predictions from {TEST_SPLIT_PATH}...")
        df_test = pd.read_csv(TEST_SPLIT_PATH)
        engine = HybridFusionEngine()
        records = []
        for _, row in df_test.head(500).iterrows():
            flow_dict = row.to_dict()
            rec = engine.process_flow(flow_dict)
            records.append(rec)
        HybridFusionEngine.persist_fusion_records(records, db_path)
        print(f"[+] Successfully seeded {len(records)} hybrid ML predictions.")

    conn.close()
    return True


if __name__ == "__main__":
    seed_benchmark_data()

