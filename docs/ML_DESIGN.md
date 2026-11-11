# Machine Learning Layer Design Document (Hybrid IDS)

## 1. Executive Summary & Design Goals
The goal of this Machine Learning layer is to turn the IDS into a genuine **Hybrid Intrusion Detection System** by combining:
1. **Deterministic Rule Engine (Suricata 8.0.7)**: Zero false positives for known signatures, strict protocol validation, and deterministic explainability strings.
2. **Behavioral Machine Learning Engine**: Anomaly detection and statistical pattern recognition for zero-day variants, slow-and-low stealth attacks, and behavioral sweeps that evade explicit signature thresholds.

### Core Architectural Invariants
- **Deterministic Explainability**: Every signature detection continues to provide an exact rule-evidence string citing observed byte/packet counts against rule thresholds. ML detections provide feature attribution (e.g., top contributing flow features and percentiles).
- **High Throughput & Zero Packet Drops**: Suricata processes live or replayed traffic without Python sitting in the inline packet-processing path. Suricata writes asynchronous `flow` records to `logs/eve.json`. The ML ingestion pipeline processes completed flow records out-of-band without degrading capture performance.
- **Training & Runtime Parity**: Machine learning features are strictly derived from Suricata EVE `flow` records. No feature can be used during training that cannot be computed directly from an EVE `flow` JSON record at runtime.

---

## 2. Feature Specification (Derived from Suricata EVE `flow` Events)

Suricata EVE `flow` events are emitted when a flow terminates or times out. A representative EVE `flow` event structure:
```json
{
  "timestamp": "2017-07-06T13:20:05.123456+00:00",
  "flow_id": 123456789012345,
  "event_type": "flow",
  "src_ip": "172.16.0.1",
  "src_port": 49152,
  "dest_ip": "192.168.10.50",
  "dest_port": 80,
  "proto": "TCP",
  "app_proto": "http",
  "flow": {
    "pkts_toserver": 15,
    "pkts_toclient": 12,
    "bytes_toserver": 1850,
    "bytes_toclient": 4200,
    "start": "2017-07-06T13:20:00.000000+00:00",
    "end": "2017-07-06T13:20:05.000000+00:00",
    "age": 5,
    "state": "established",
    "reason": "shutdown"
  },
  "tcp": {
    "tcp_flags": "1b",
    "tcp_flags_ts": "12",
    "tcp_flags_tc": "1b",
    "syn": true,
    "fin": true,
    "ack": true,
    "psh": true,
    "rst": false
  }
}
```

### Shared Feature Vector (`app/ml/features.py`)
Both training data preparation and live inference utilize the exact same feature extraction function:

| # | Feature Name | Source Field in EVE `flow` | Description / Formula |
|---|---|---|---|
| 1 | `duration` | `flow.age` or `(end - start)` | Flow duration in seconds: `max(age, 0.0001)` |
| 2 | `pkts_toserver` | `flow.pkts_toserver` | Packets sent from source to destination |
| 3 | `pkts_toclient` | `flow.pkts_toclient` | Packets sent from destination to source |
| 4 | `total_pkts` | `pkts_toserver + pkts_toclient` | Total bidirectional packet count |
| 5 | `bytes_toserver` | `flow.bytes_toserver` | Total bytes sent to server |
| 6 | `bytes_toclient` | `flow.bytes_toclient` | Total bytes sent to client |
| 7 | `total_bytes` | `bytes_toserver + bytes_toclient` | Total bidirectional byte volume |
| 8 | `bytes_per_sec` | `total_bytes / duration` | Data transfer rate in bytes per second |
| 9 | `pkts_per_sec` | `total_pkts / duration` | Packet rate in packets per second |
| 10 | `bytes_ratio` | `bytes_toserver / max(bytes_toclient, 1)` | Ratio of outgoing to incoming payload |
| 11 | `pkts_ratio` | `pkts_toserver / max(pkts_toclient, 1)` | Ratio of client packets to server packets |
| 12 | `avg_pkt_size` | `total_bytes / max(total_pkts, 1)` | Mean packet size across the flow |
| 13 | `avg_pkt_size_toserver` | `bytes_toserver / max(pkts_toserver, 1)` | Mean packet size to server |
| 14 | `avg_pkt_size_toclient` | `bytes_toclient / max(pkts_toclient, 1)` | Mean packet size to client |
| 15 | `dest_port` | `dest_port` | Numerical destination port |
| 16 | `dest_port_is_privileged`| `dest_port < 1024` | Binary flag indicating standard service port |
| 17 | `proto_num` | `proto` | Encoded protocol (TCP=6, UDP=17, ICMP=1, Other=0) |
| 18 | `app_proto_code` | `app_proto` | Categorical encoding (http, tls, ssh, ftp, dns, etc.) |
| 19 | `flow_state_code`| `flow.state` | Encoded state (new=1, established=2, closed=3) |
| 20 | `flow_reason_code`| `flow.reason` | Encoded closure reason (timeout=1, shutdown=2, other=0)|
| 21 | `tcp_flag_syn` | `tcp.syn` | Binary: SYN flag observed |
| 22 | `tcp_flag_fin` | `tcp.fin` | Binary: FIN flag observed |
| 23 | `tcp_flag_rst` | `tcp.rst` | Binary: RST flag observed |
| 24 | `tcp_flag_psh` | `tcp.psh` | Binary: PSH flag observed |
| 25 | `tcp_flag_ack` | `tcp.ack` | Binary: ACK flag observed |

---

## 3. SQLite Schema Additions

To support hybrid telemetry without degrading write throughput:
1. Enable **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` for concurrent reading and writing.
2. Add the `ml_predictions` table to store inference results and explanations.

```sql
-- Enable WAL mode for high concurrency
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

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
    is_anomaly INTEGER NOT NULL,          -- 0 = Benign, 1 = Attack/Anomaly
    confidence REAL NOT NULL,              -- Predicted class probability [0.0 - 1.0]
    top_contributing_features TEXT NOT NULL, -- JSON string of top SHAP / feature values
    fusion_status TEXT NOT NULL,           -- 'CORROBORATED', 'ANOMALY_ONLY', 'SIGNATURE_ONLY', 'BENIGN'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_pred_5tuple ON ml_predictions(src_ip, src_port, dest_ip, dest_port, proto);
CREATE INDEX IF NOT EXISTS idx_ml_pred_fusion ON ml_predictions(fusion_status);
CREATE INDEX IF NOT EXISTS idx_ml_pred_time ON ml_predictions(timestamp);
```

---

## 4. Hybrid Fusion Logic & Decision Matrix

When a flow record concludes, the system queries whether a Suricata alert matched the same session within a matching temporal window:

```
                                  [Flow Record Completed]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [Suricata Alert Triggered?]               [ML Classifier Inference]
                       │                                           │
         ┌─────────────┴─────────────┐               ┌─────────────┴─────────────┐
         ▼                           ▼               ▼                           ▼
       YES                          NO              ATTACK                     BENIGN
```

### Fusion Decision Matrix

| Suricata Signature Alert | ML Classification | Fusion Status | Final Action & Explainability |
|---|---|---|---|
| **YES** (SID matched) | **ATTACK** (Corroborating) | `CORROBORATED` | **Priority 1 Alert**: High-confidence attack. Displays rule explainability string **plus** ML feature attribution explaining behavioral characteristics (e.g. `pkts_per_sec` and `bytes_ratio`). |
| **YES** (SID matched) | **BENIGN** (Low confidence) | `SIGNATURE_ONLY` | **Priority 2 Alert**: Deterministic rule takes precedence (ensuring no security regression), but flagged with note: *"ML assessed baseline behavioral parameters; rule condition met"*. |
| **NO** (No rule triggered)| **ATTACK** (Anomaly/High Conf)| `ANOMALY_ONLY` | **Priority 2 / Investigation Alert**: Potential zero-day, evasion attempt, or stealth scan. Fully explained via top anomalous features (e.g. `syn_ratio=1.0, duration=0.01s, dest_port=22`). |
| **NO** (No rule triggered)| **BENIGN** | `BENIGN` | **Suppressed**: Clean flow verified by both signature rules and behavioral model. No alert generated. |

---

## 5. Throughput & Zero-Packet-Drop Architecture

To guarantee zero packet drops:
1. **Zero Python in the Inline Fast Path**:
   - Suricata processes packets at multi-gigabit line rates via AF_PACKET / DPDK worker threads.
   - Suricata's logging thread dumps JSON records into `logs/eve.json`.
2. **Asynchronous Non-Blocking Tailer**:
   - `LiveEveTailer` consumes `flow` and `alert` events in micro-batches (50–100 records).
   - Feature extraction operates on pre-allocated NumPy / native Python dictionaries, requiring `< 0.05 ms` per record.
   - Inference uses vectorized batch processing in LightGBM/ONNX runtime or scikit-learn models.
   - Database operations use WAL mode with batched `executemany` commits to prevent database write locks from stalling the pipeline.

---

## 6. Data Cleaning, Known Dataset Caveats & Time-Based Splits

### Dataset Integrity & Splitting Protocol
To eliminate data leakage and spatial correlation:
- **No Random Row-Level Splits**: Random splits leak flows from identical TCP sessions across train and test sets.
- **Strict Time/Day-Based Split**:
  - **Training Set**: Monday (Benign baseline) + Tuesday (Brute Force) + Wednesday (DoS / Heartbleed).
  - **Validation Set**: Held-out chronological slice of Wednesday traffic.
  - **Testing Set**: Thursday (Web Attacks, Infiltration) + Friday (Botnet, PortScan, DDoS).

### Known CIC-IDS2017 Dataset Quirks & Cleaning Procedures
1. **Inf and NaN Values**: Flow duration = 0 causing division-by-zero errors in flow rates (`Flow Bytes/s`, `Flow Packets/s`).
   - *Fix*: Apply `max(duration, 0.0001)` and cap rate metrics at the 99.9th percentile.
2. **Duplicate Rows**: Replayed PCAP fragments in CIC-IDS2017 contain duplicate flow records.
   - *Fix*: Deduplicate on `(5-tuple, timestamp, flow_duration)`.
3. **Mislabeled Flows**: Documented label errors in Wednesday/Thursday morning captures where normal web browsing was tagged as attack or vice-versa.
   - *Fix*: Standardize labels using normalized categories and exclude ambiguous background ICMP broadcast packets.

---

## 7. Model Selection & Evaluation Protocol

### Model Architecture
- **Primary Model**: LightGBM Classifier (fastest CPU inference, native handling of categorical features, low memory footprint).
- **Baseline Comparison**: Random Forest / Isolation Forest for unsupervised anomaly comparison.

### Evaluation Metrics (Accuracy Never Reported Alone)
Because intrusion detection datasets are heavily imbalanced (>80% Benign), accuracy is misleading. Evaluation must report:
1. **Per-Class Precision, Recall, and F1-Score**.
2. **Precision-Recall AUC (PR-AUC)** per class.
3. **Multi-Class Confusion Matrix** (Raw counts and normalized proportions).
4. **False Alarm Rate (FAR)** on Benign traffic.
5. **Inference Latency**: Microseconds per flow at runtime.

---

## 8. Implementation Roadmap

- **Phase 0**: Repo audit & Design Document (`docs/ML_DESIGN.md`) [CURRENT]
- **Phase 1**: Feature extractor (`app/ml/features.py`) with unit tests (`tests/test_features.py`)
- **Phase 2**: Data preparation & Time-based splitting pipeline (`scripts/prepare_dataset.py`)
- **Phase 3**: Model training & Evaluation reporting (`app/ml/train.py`, `app/ml/evaluate.py`)
- **Phase 4**: Hybrid Fusion Engine & EVE Tailer integration (`app/ml/fusion.py`, `src/eve_ingestor.py`)
- **Phase 5**: API endpoints & Dashboard visualization updates (`app/server.py`, `app/static/`)
