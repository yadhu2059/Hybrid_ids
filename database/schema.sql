-- SQLite DDL Schema for Hybrid Signature & Machine Learning IDS Engine

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    src_ip TEXT NOT NULL,
    src_port INTEGER NOT NULL,
    dest_ip TEXT NOT NULL,
    dest_port INTEGER NOT NULL,
    proto TEXT NOT NULL,
    signature_id INTEGER NOT NULL,
    signature TEXT NOT NULL,
    category TEXT,
    severity INTEGER NOT NULL,
    bytes_toserver INTEGER DEFAULT 0,
    pkts_toserver INTEGER DEFAULT 0,
    bytes_toclient INTEGER DEFAULT 0,
    pkts_toclient INTEGER DEFAULT 0,
    reason TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_5tuple ON alerts(src_ip, src_port, dest_ip, dest_port, proto);
CREATE INDEX IF NOT EXISTS idx_alerts_sid ON alerts(signature_id);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

CREATE TABLE IF NOT EXISTS evaluation_flows (
    flow_id TEXT PRIMARY KEY,
    src_ip TEXT NOT NULL,
    src_port INTEGER NOT NULL,
    dest_ip TEXT NOT NULL,
    dest_port INTEGER NOT NULL,
    proto TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration REAL DEFAULT 0.0,
    ground_truth_label TEXT NOT NULL,
    expected_sid INTEGER,
    detected_sid INTEGER,
    detected_signature TEXT,
    match_status TEXT NOT NULL, -- 'TP', 'FP', 'TN', 'FN'
    overlapping_sids TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eval_status ON evaluation_flows(match_status);
CREATE INDEX IF NOT EXISTS idx_eval_label ON evaluation_flows(ground_truth_label);
CREATE INDEX IF NOT EXISTS idx_eval_detected_sid ON evaluation_flows(detected_sid);

CREATE TABLE IF NOT EXISTS benchmark_metrics (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    total_runtime_sec REAL NOT NULL,
    pcap_size_bytes INTEGER NOT NULL,
    total_flows INTEGER NOT NULL,
    total_packets INTEGER NOT NULL,
    throughput_mb_s REAL NOT NULL,
    throughput_flows_s REAL NOT NULL,
    peak_memory_rss_mb REAL NOT NULL,
    total_alerts INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    src_ip TEXT NOT NULL,
    src_port INTEGER NOT NULL,
    dest_ip TEXT NOT NULL,
    dest_port INTEGER NOT NULL,
    proto TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    is_anomaly INTEGER NOT NULL,
    confidence REAL NOT NULL,
    attack_probability REAL NOT NULL,
    top_contributing_features TEXT NOT NULL, -- JSON formatted list of top features
    fusion_status TEXT NOT NULL,           -- 'CORROBORATED', 'ANOMALY_ONLY', 'SIGNATURE_ONLY', 'BENIGN'
    signature_id INTEGER DEFAULT 0,
    signature TEXT DEFAULT '',
    explainability_summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_pred_5tuple ON ml_predictions(src_ip, src_port, dest_ip, dest_port, proto);
CREATE INDEX IF NOT EXISTS idx_ml_pred_fusion ON ml_predictions(fusion_status);
CREATE INDEX IF NOT EXISTS idx_ml_pred_time ON ml_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_pred_label ON ml_predictions(predicted_label);
