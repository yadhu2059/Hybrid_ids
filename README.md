# 🛡️ Hybrid Intrusion Detection & Attribution System (H-IDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![Suricata](https://img.shields.io/badge/Suricata-8.0.7-brightgreen.svg)](https://suricata.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance **Hybrid Network Intrusion Detection & Explainable Threat Attribution Engine** built on genuine **Suricata 8.0.7 telemetry**, deterministic rule engines, real-time packet stream decoding, and automated 5-tuple ground-truth flow evaluation against standardized benchmarks (e.g., **CICIDS2017**).

---

## 📌 Architectural Overview

```mermaid
graph TD
    A[Raw PCAP / Network Stream] -->|PCAP Replay / Live Interface| B(Suricata 8.0.7 Engine)
    B -->|Custom Rules Config: SIDs 1000001-1000014| C{Inspection Engine}
    C -->|Real-Time Telemetry| D[logs/eve.json Log Stream]
    D -->|LiveEveTailer Pipeline| E[(database/alerts.db SQLite)]
    E -->|Server-Sent Events SSE| F[FastAPI Backend Server]
    F -->|Real-Time Telemetry & SSE| G[Responsive Dark-Mode SPA Dashboard]
    
    H[Ground-Truth Flows: ground_truth.csv] --> I[5-Tuple Flow Alignment Engine]
    E --> I
    I -->|Conflict Resolution & Attribution| J[Detection Metrics & Confusion Matrix]
    J --> G
```

---

## ✨ Key Features

1. **Deterministic Rule Engine & Zero-Noise Calibration**:
   - 14 customized and tuned Suricata detection rules covering DoS/DDoS (LOIC, TCP SYN, Slowloris, SlowHTTPTest, Hulk), Web Application Attacks (SQLi, XSS, HTTP Brute Force), Scanning Probing (SYN scan, connect sweeps), and Lateral Infiltration.
   - Calibrated detection filters and threshold state tables to maintain **0% False Alarm Rate (FAR)** across verified benign traffic.

2. **Strict 5-Tuple Ground-Truth Evaluation Engine**:
   - Mathematical 5-tuple flow alignment (Source IP, Source Port, Dest IP, Dest Port, Protocol, Timestamp) within a dynamic temporal window (+/- 2.0s).
   - Explicit **Misclassification Tracking**: Differentiates between actual True Positives and cross-category signature misclassifications.
   - Complete statistical rigor: Precision, Recall, F1-Score, and Full Multi-Class Confusion Matrix calculated live with zero hardcoding.

3. **High-Throughput Live Ingestion & Replay**:
   - Subprocess pipeline executing Suricata in live replay mode over gigabyte-scale PCAPs (e.g., CICIDS2017 `Thursday-WorkingHours.pcap` with 9.3M+ packets).
   - Real-time `eve.json` tailing writing into an indexed SQLite store.

4. **Modern Dark-Mode Operations Dashboard**:
   - Single-Page Web Application powered by **FastAPI** and modern vanilla JavaScript / CSS (zero external Node.js dependencies).
   - Real-Time Server-Sent Events (SSE) telemetry progress bars, live notification feeds, interactive Plotly donut distribution charts, and expand/collapse threat deduplication cards.
   - Integrated **Attack Simulator** for on-the-fly packet synthesis and live signature firing verification.
   - Clean Database Reset and session state management controls.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python 3.10+**
- **Suricata 7.x / 8.x** (or **Docker** for containerized execution)
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/Sreehariiiii/Hybrid-Intrusion-Detection-System.git
cd Hybrid-Intrusion-Detection-System
```

### 3. Install Dependencies
```bash
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate

pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```
Open your browser at: **http://localhost:8000**

---

## 🔬 Benchmark Dataset Evaluation (CICIDS2017 Thursday)

Evaluated across **6,216 ground-truth network flows** containing Web Attacks, Infiltration, and 4,000 Benign background flows:

| Category | Flows | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Misclassified | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 4,000 | 4,000 *(TN)* | 0 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| **Web Attack - SQL Injection** | 21 | 21 | 0 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| **Web Attack - Brute Force** | 1,507 | 1,503 | 0 | 0 | 4 | **99.87%** | **99.73%** | **99.80%** |
| **Web Attack - XSS** | 652 | 649 | 0 | 0 | 3 | **99.54%** | **99.54%** | **99.54%** |
| **Infiltration** | 36 | 18 | 0 | 0 | 18 | **100.0%** | **50.00%** | **66.67%** |

* **Overall Exact-Class Accuracy**: `99.60%`
* **Behavioural Attack Detection Rate (Any Attack Alert)**: `100.00%` (2,216 / 2,216 attack flows flagged)
* **False Alarm Rate (Benign Traffic)**: `0.00%` (0 false positives on 4,000 benign flows)

> **Note**: Ground-truth benchmark evaluation on 6,216 flows is separated from the full PCAP operational analysis (which processes the full 9.3M+ packet `Thursday-WorkingHours.pcap` to evaluate line-rate throughput and real-world system resilience).

---

## 🧠 Machine Learning Layer & Hybrid Fusion Engine

The machine learning layer makes the **"Hybrid"** in Hybrid-IDS real by combining deterministic signature detection with behavioral flow anomaly classification while preserving:
- **Deterministic Explainability**: Every alert includes exact rule conditions or ML feature attribution $z$-scores.
- **High Throughput**: **$8.33\ \mu\text{s}$ per flow** CPU inference latency ($> 120,000$ flows/s).
- **Zero Packet Drops**: Completed flows are tailed asynchronously out-of-band without blocking Suricata's capture threads.
- **Zero Data Leakage**: Enforces strict chronological time-based splitting (train Mon–Wed, test Thu–Fri).

### 25-Feature Shared Extraction Vector (`app/ml/features.py`)
Features are derived strictly from Suricata EVE `flow` records with 100% training/runtime parity:
1. `duration`, `pkts_toserver`, `pkts_toclient`, `total_pkts`, `bytes_toserver`, `bytes_toclient`, `total_bytes`
2. `bytes_per_sec`, `pkts_per_sec`, `bytes_ratio`, `pkts_ratio`
3. `avg_pkt_size`, `avg_pkt_size_toserver`, `avg_pkt_size_toclient`
4. `dest_port`, `dest_port_is_privileged` ($< 1024$), `proto_num`, `app_proto_code`
5. `flow_state_code`, `flow_reason_code`
6. `tcp_flag_syn`, `tcp_flag_fin`, `tcp_flag_rst`, `tcp_flag_psh`, `tcp_flag_ack`

### Hybrid Fusion Arbitration Matrix (`app/ml/fusion.py`)
- **`CORROBORATED`**: Signature matched + ML attack $\rightarrow$ Elevated Priority 1 Alert with dual explanation.
- **`SIGNATURE_ONLY`**: Signature matched + ML benign $\rightarrow$ Deterministic rule takes precedence (no security regression).
- **`ANOMALY_ONLY`**: Signature silent + ML anomaly $\rightarrow$ Zero-day or evasion suspect with top-3 anomalous feature z-scores.
- **`BENIGN`**: Signature silent + ML baseline $\rightarrow$ Verified clean flow.

### Running ML Pipelines & Tests

```bash
# 1. Run complete unit test suite (31 tests)
python -m pytest tests/

# 2. Extract ML features from Suricata EVE log
python scripts/extract_features.py --input logs/eve.json --output data/extracted_features.csv

# 3. Clean and split dataset with strict chronological time-based partitions
python scripts/prepare_dataset.py --input data/ground_truth.csv --output-dir data/splits

# 4. Train LightGBM model and generate scientific evaluation report
python -m app.ml.train
```

---

## 📂 Project Structure

```
├── app/
│   ├── server.py              # FastAPI application server, SSE event hub & REST API
│   ├── ml/                    # Machine Learning subsystem
│   │   ├── features.py        # Shared 25-feature extractor (100% runtime/training parity)
│   │   ├── dataset.py         # Data cleaning, deduplication & time-based splitting
│   │   ├── model.py           # LightGBM classifier & statistical feature-attribution engine
│   │   ├── evaluate.py        # Scientific evaluation (per-class PR-AUC, F1, confusion matrix)
│   │   ├── train.py           # Model training and artifact persistence pipeline
│   │   └── fusion.py          # Hybrid Fusion Engine & SQLite WAL persistence
│   └── static/
│       ├── index.html         # Responsive operations dashboard SPA with Hybrid ML tab
│       ├── styles.css         # Modern dark-mode UI stylesheet
│       └── app.js             # Client-side reactivity, Plotly charts & SSE listener
├── config/
│   ├── custom_rules.rules     # Calibrated Suricata detection rules (SIDs 1000001-1000014)
│   └── suricata.yaml          # Suricata engine runtime configuration
├── data/
│   ├── ground_truth.csv       # Extracted benchmark flow ground-truth labels
│   └── splits/                # Chronologically split train, val, and test partitions
├── database/
│   ├── alerts.db              # SQLite storage for live alerts, evaluations & ML predictions
│   └── schema.sql             # Relational database schema with WAL mode pragmas
├── docs/
│   └── ML_DESIGN.md           # Machine learning layer design & specification document
├── models/
│   ├── hybrid_model.joblib    # Persisted LightGBM model artifact
│   └── evaluation_report.json # Scientific evaluation metrics
├── scripts/
│   ├── extract_features.py    # CLI feature extractor
│   └── prepare_dataset.py     # CLI dataset cleaner & chronological splitter
├── src/
│   ├── alert_notifier.py      # Rule complexity parser and webhook notification dispatcher
│   ├── attack_injector.py     # Scapy attack synthesis and injection module
│   ├── benchmark.py           # Throughput and RSS memory benchmark engine
│   ├── evaluator.py           # 5-tuple flow matching, attribution, and confusion matrix engine
│   └── eve_ingestor.py        # Live Suricata subprocess runner & eve.json tailer (Alert + Flow)
├── tests/
│   ├── test_features.py       # Unit tests for shared feature extractor
│   ├── test_dataset.py        # Unit tests for dataset cleaning and chronological splits
│   ├── test_model.py          # Unit tests for model training and inference
│   ├── test_fusion.py         # Unit tests for hybrid decision fusion logic
│   └── test_api.py            # Unit tests for FastAPI REST endpoints
├── Dockerfile                 # Docker container definition
├── requirements.txt           # Python dependency manifest
└── README.md                  # System documentation & reference
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
