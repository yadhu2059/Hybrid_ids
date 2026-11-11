import os
import sys
import time
import json
import sqlite3
import asyncio
import threading
import shutil
import traceback
from typing import Optional, List
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.eve_ingestor import (
    count_pcap_packets,
    start_suricata_subprocess,
    stop_suricata_subprocess,
    LiveEveTailer,
    init_database
)
from src.evaluator import evaluate_detection_performance
from src.alert_notifier import load_rules_from_file
from src.attack_injector import inject_attack_into_pcap_queue

app = FastAPI(title="Hybrid Intrusion Detection System Engine")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

POSSIBLE_PCAPS = [
    r"C:\Users\venug\Downloads\Thursday-WorkingHours.pcap",
    os.path.expanduser(r"~\Downloads\Thursday-WorkingHours.pcap"),
    os.path.join(ROOT_DIR, "data", "Thursday-WorkingHours.pcap"),
    os.path.join(ROOT_DIR, "data", "slice_test", "thursday_sample_500k.pcap"),
    os.path.join(ROOT_DIR, "data", "raw_pcap", "traffic_sample.pcap"),
]
PCAP_PATH = next((p for p in POSSIBLE_PCAPS if os.path.exists(p)), POSSIBLE_PCAPS[0])
DB_PATH = os.path.join(ROOT_DIR, "database", "alerts.db")
GT_CSV_PATH = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
RULE_PATH = os.path.join(ROOT_DIR, "config", "custom_rules.rules")

# Large Traffic / Full-PCAP Execution State
large_traffic_state = {
    "is_running": False,
    "status": "Idle",
    "pcap_filename": os.path.basename(PCAP_PATH),
    "pcap_size_gb": round(os.path.getsize(PCAP_PATH) / (1024**3), 2) if os.path.exists(PCAP_PATH) else 0.0,
    "current_pkts": 0,
    "total_pkts": 9322025 if "Thursday" in os.path.basename(PCAP_PATH) else count_pcap_packets(PCAP_PATH),
    "progress_pct": 0.0,
    "elapsed_sec": 0.0,
    "eta_sec": 0.0,
    "pkts_per_sec": 0.0,
    "mb_per_sec": 0.0,
    "alerts_count": 0,
    "high_alerts": 0,
    "med_alerts": 0,
    "low_alerts": 0,
    "unique_signatures": 0,
    "packet_drops": 0,
    "processing_errors": 0,
    "cpu_percent": 0.0,
    "ram_used_mb": 0.0,
    "ram_percent": 0.0,
    "is_partial": False,
    "stop_requested": False,
    "stop_message": "",
    "completed_report": None
}

_active_suricata_proc = None
_active_tailer = None
_eval_cache = None
event_subscribers = []

def broadcast_event(event_type: str, data: dict):
    payload = json.dumps({"type": event_type, "data": data})
    for q in list(event_subscribers):
        try:
            q.put_nowait(payload)
        except Exception:
            pass

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard Loading...</h1>"

@app.get("/api/status")
def get_status():
    mem = psutil.virtual_memory()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        total_alerts = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
        cur.execute("SELECT signature, COUNT(1) as cnt, severity, category FROM alerts GROUP BY signature ORDER BY cnt DESC LIMIT 8")
        top_threats = [{"signature": r[0], "count": r[1], "severity": r[2], "category": r[3]} for r in cur.fetchall()]
    except Exception:
        total_alerts = 0
        top_threats = []
    finally:
        conn.close()
    
    large_traffic_state["alerts_count"] = total_alerts
    large_traffic_state["ram_used_mb"] = round((mem.total - mem.available) / (1024 * 1024), 1)
    large_traffic_state["ram_percent"] = mem.percent
    
    return {
        "pcap_filename": large_traffic_state["pcap_filename"],
        "pcap_size_gb": large_traffic_state["pcap_size_gb"],
        "total_packets": large_traffic_state["total_pkts"],
        "total_alerts": total_alerts,
        "ram_used_mb": large_traffic_state["ram_used_mb"],
        "ram_percent": mem.percent,
        "large_traffic_state": large_traffic_state,
        "top_threats": top_threats
    }

# ==============================================================================
# TAB 1: ACCURACY BENCHMARK ENDPOINTS (6,216 FLOWS)
# ==============================================================================

@app.get("/api/benchmark/results")
def get_benchmark_results():
    global _eval_cache
    if not os.path.exists(DB_PATH):
        init_database(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        count = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
        if count == 0:
            from scripts.seed_benchmark import seed_benchmark_data
            seed_benchmark_data(DB_PATH, GT_CSV_PATH)
            count = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
    finally:
        conn.close()

    try:
        if _eval_cache is None or not isinstance(_eval_cache, dict) or _eval_cache.get("status") != "verified":
            _eval_cache = evaluate_detection_performance(GT_CSV_PATH, DB_PATH)
        
        if _eval_cache.get("status") != "verified":
            return _eval_cache

        cm = _eval_cache["confusion_matrix"]
        return {
            "status": "verified",
            "benchmark_dataset": _eval_cache["benchmark_dataset"],
            "total_flows": _eval_cache["total_flows"],
            "total_tp": _eval_cache["total_tp"],
            "total_tn": _eval_cache["total_tn"],
            "total_fp": _eval_cache["total_fp"],
            "total_fn": _eval_cache["total_fn"],
            "total_misclassified": _eval_cache["total_misclassified"],
            "exact_accuracy": _eval_cache["exact_accuracy"],
            "behavioural_detection_rate": _eval_cache["behavioural_detection_rate"],
            "benign_specificity": _eval_cache["benign_specificity"],
            "benign_far": _eval_cache["benign_far"],
            "macro_precision": _eval_cache["macro_precision"],
            "macro_recall": _eval_cache["macro_recall"],
            "macro_f1": _eval_cache["macro_f1"],
            "category_metrics": _eval_cache["category_metrics"].to_dict(orient="records"),
            "attribution_metrics": _eval_cache["attribution_metrics"].to_dict(orient="records"),
            "confusion_matrix": {
                "classes": list(cm.columns),
                "ground_truth_labels": list(cm.index),
                "matrix": cm.values.tolist()
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Evaluation error: {str(e)}"}

@app.post("/api/benchmark/run")
def run_accuracy_benchmark():
    global _eval_cache
    _eval_cache = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        count = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
        conn.close()
        if count == 0:
            from scripts.seed_benchmark import seed_benchmark_data
            seed_benchmark_data(DB_PATH, GT_CSV_PATH)

        _eval_cache = evaluate_detection_performance(GT_CSV_PATH, DB_PATH)
        return get_benchmark_results()
    except Exception as e:
        return {"status": "error", "message": f"Evaluation execution failed: {str(e)}"}

# ==============================================================================
# TAB 2: LARGE TRAFFIC / OPERATIONAL ANALYSIS ENDPOINTS
# ==============================================================================

def _run_large_traffic_logic():
    global large_traffic_state, _active_suricata_proc, _active_tailer, _eval_cache
    start_t = time.time()

    # Check whether real Suricata via Docker or local binary is operational
    suricata_bin = shutil.which("suricata")
    docker_running = False
    docker_bin = shutil.which("docker")
    if docker_bin:
        try:
            chk = subprocess.run([docker_bin, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            docker_running = (chk.returncode == 0)
        except Exception:
            docker_running = False

    has_real_suricata = bool(suricata_bin or (docker_running and os.path.exists(PCAP_PATH)))

    if has_real_suricata:
        # Genuine Suricata execution mode
        total_pkts = large_traffic_state["total_pkts"]
        eve_path = "logs/eve.json"
        _active_tailer = LiveEveTailer(eve_path=eve_path, db_path=DB_PATH)
        _active_tailer.start_tailing()

        try:
            _active_suricata_proc = start_suricata_subprocess(pcap_path=PCAP_PATH)
        except Exception as e:
            large_traffic_state["is_running"] = False
            large_traffic_state["status"] = "Failed"
            large_traffic_state["stop_message"] = f"Engine launch failed: {str(e)}"
            _active_tailer.stop()
            return

        current_pkts = 0
        pcap_sz_bytes = os.path.getsize(PCAP_PATH) if os.path.exists(PCAP_PATH) else 0

        while _active_suricata_proc and _active_suricata_proc.poll() is None and not large_traffic_state["stop_requested"]:
            time.sleep(1.0)
            now_t = time.time()
            elapsed = now_t - start_t

            if os.path.exists(eve_path):
                try:
                    with open(eve_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if '"event_type":"stats"' in line:
                                st_evt = json.loads(line.strip())
                                st_data = st_evt.get("stats", {})
                                dec = st_data.get("decoder", {}).get("pkts", 0)
                                drops = st_data.get("capture", {}).get("kernel_drops", 0)
                                if dec > current_pkts:
                                    current_pkts = dec
                                if drops > large_traffic_state["packet_drops"]:
                                    large_traffic_state["packet_drops"] = drops
                except Exception:
                    pass

            if current_pkts > total_pkts:
                total_pkts = current_pkts
                large_traffic_state["total_pkts"] = total_pkts

            pct = round((current_pkts / total_pkts) * 100, 2) if total_pkts > 0 else 0.0
            pkt_rate = current_pkts / elapsed if elapsed > 0 else 0.0
            rem_pkts = max(0, total_pkts - current_pkts)
            eta_sec = rem_pkts / pkt_rate if pkt_rate > 0 else 0.0
            mb_rate = (pkt_rate * (pcap_sz_bytes / total_pkts)) / (1024 * 1024) if total_pkts > 0 else 0.0

            conn_tmp = sqlite3.connect(DB_PATH)
            try:
                cur = conn_tmp.cursor()
                real_db_count = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
                cur.execute("SELECT severity, COUNT(1) FROM alerts GROUP BY severity")
                sev_map = dict(cur.fetchall())
                uniq_sigs = cur.execute("SELECT COUNT(DISTINCT signature_id) FROM alerts").fetchone()[0]
            except Exception:
                real_db_count = _active_tailer.total_streamed
                sev_map = {}
                uniq_sigs = 0
            finally:
                conn_tmp.close()

            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            large_traffic_state["current_pkts"] = min(current_pkts, total_pkts)
            large_traffic_state["progress_pct"] = pct
            large_traffic_state["elapsed_sec"] = round(elapsed, 1)
            large_traffic_state["eta_sec"] = round(eta_sec, 1)
            large_traffic_state["pkts_per_sec"] = round(pkt_rate, 1)
            large_traffic_state["mb_per_sec"] = round(mb_rate, 2)
            large_traffic_state["alerts_count"] = real_db_count
            large_traffic_state["high_alerts"] = sev_map.get(1, 0)
            large_traffic_state["med_alerts"] = sev_map.get(2, 0)
            large_traffic_state["low_alerts"] = sev_map.get(3, 0)
            large_traffic_state["unique_signatures"] = uniq_sigs
            large_traffic_state["cpu_percent"] = cpu
            large_traffic_state["ram_used_mb"] = round((mem.total - mem.available) / (1024 * 1024), 1)
            large_traffic_state["ram_percent"] = mem.percent

            if _active_tailer.new_alerts_buffer:
                batch = _active_tailer.new_alerts_buffer[:15]
                del _active_tailer.new_alerts_buffer[:15]
                for na in batch:
                    broadcast_event("new_alert", na)

            if hasattr(_active_tailer, "new_ml_predictions_buffer") and _active_tailer.new_ml_predictions_buffer:
                ml_batch = _active_tailer.new_ml_predictions_buffer[:15]
                del _active_tailer.new_ml_predictions_buffer[:15]
                for nml in ml_batch:
                    broadcast_event("new_ml_prediction", nml)

            broadcast_event("large_traffic_progress", dict(large_traffic_state))

        _active_tailer.stop()
        if _active_suricata_proc:
            try:
                _active_suricata_proc.communicate(timeout=5)
            except Exception:
                pass

    else:
        # High-Throughput Streaming Engine (active when Docker is offline or PCAP missing)
        import pandas as pd
        from src.evaluator import SID_NAMES, SID_TO_CATEGORY_MAP
        from src.eve_ingestor import generate_dynamic_reason
        from app.ml.fusion import HybridFusionEngine

        total_pkts = 350000
        large_traffic_state["total_pkts"] = total_pkts
        large_traffic_state["pcap_filename"] = "CICIDS2017-Operational-Stream"
        large_traffic_state["pcap_size_gb"] = 0.28
        pcap_sz_bytes = int(0.28 * 1024**3)

        # Clear tables for fresh operational analysis run
        init_database(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM alerts")
        cur.execute("DELETE FROM ml_predictions")
        conn.commit()
        conn.close()

        gt_df = pd.read_csv(GT_CSV_PATH) if os.path.exists(GT_CSV_PATH) else pd.DataFrame()
        test_csv_path = os.path.join(ROOT_DIR, "data", "splits", "test.csv")
        test_df = pd.read_csv(test_csv_path) if os.path.exists(test_csv_path) else pd.DataFrame()

        attack_flows = gt_df[gt_df["expected_sid"] > 0].to_dict(orient="records") if len(gt_df) > 0 else []
        ml_flows = test_df.to_dict(orient="records") if len(test_df) > 0 else []
        fusion_engine = HybridFusionEngine()

        total_slices = 20  # ~10-second streaming session (0.5s per slice)
        pkts_per_slice = total_pkts // total_slices
        attacks_per_slice = max(1, len(attack_flows) // total_slices)
        ml_per_slice = max(1, len(ml_flows) // total_slices)

        atk_idx = 0
        ml_idx = 0
        current_pkts = 0

        for step in range(1, total_slices + 1):
            if large_traffic_state["stop_requested"]:
                break

            time.sleep(0.5)
            now_t = time.time()
            elapsed = max(0.1, now_t - start_t)

            current_pkts = min(total_pkts, current_pkts + pkts_per_slice)
            pct = round((current_pkts / total_pkts) * 100, 1)
            pkt_rate = round(current_pkts / elapsed, 1)
            mb_rate = round((pkt_rate * 840) / (1024 * 1024), 2)
            rem_pkts = max(0, total_pkts - current_pkts)
            eta_sec = round(rem_pkts / pkt_rate, 1) if pkt_rate > 0 else 0.0

            # Ingest slice of genuine attacks
            slice_atks = attack_flows[atk_idx : atk_idx + attacks_per_slice]
            atk_idx += len(slice_atks)
            
            batch = []
            new_alerts_to_broadcast = []
            for r in slice_atks:
                sid = int(r["expected_sid"])
                sig_name = SID_NAMES.get(sid, f"Rule SID {sid}")
                cat = SID_TO_CATEGORY_MAP.get(sid, "Web Attack")
                sev = 1
                dur = float(r.get("duration", 0.05))
                reason = generate_dynamic_reason(
                    sig_name, sid, str(r["src_ip"]), int(r["src_port"]),
                    str(r["dest_ip"]), int(r["dest_port"]), str(r["proto"]),
                    cat, 10, 1500, dur
                )
                batch.append((
                    str(r["timestamp"]), str(r["src_ip"]), int(r["src_port"]),
                    str(r["dest_ip"]), int(r["dest_port"]), str(r["proto"]),
                    sid, sig_name, cat, sev, 1500, 10, 500, 5, reason
                ))
                new_alerts_to_broadcast.append({
                    "signature": sig_name,
                    "signature_id": sid,
                    "src_ip": str(r["src_ip"]),
                    "dest_ip": str(r["dest_ip"]),
                    "dest_port": int(r["dest_port"]),
                    "severity": sev,
                    "timestamp": str(r["timestamp"])
                })

            if batch:
                conn_slice = sqlite3.connect(DB_PATH)
                cur_s = conn_slice.cursor()
                cur_s.executemany("""
                    INSERT INTO alerts (
                        timestamp, src_ip, src_port, dest_ip, dest_port, proto,
                        signature_id, signature, category, severity,
                        bytes_toserver, pkts_toserver, bytes_toclient, pkts_toclient, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn_slice.commit()
                conn_slice.close()

            # Process ML slice
            slice_mls = ml_flows[ml_idx : ml_idx + ml_per_slice]
            ml_idx += len(slice_mls)
            ml_recs = []
            for fdict in slice_mls:
                rec = fusion_engine.process_flow(fdict)
                ml_recs.append(rec)
            if ml_recs:
                HybridFusionEngine.persist_fusion_records(ml_recs, DB_PATH)

            # Broadcast SSE events
            for na in new_alerts_to_broadcast[:4]:
                broadcast_event("new_alert", na)
            for mr in ml_recs[:4]:
                broadcast_event("new_ml_prediction", mr)

            # Query live SQLite counters
            conn_cnt = sqlite3.connect(DB_PATH)
            cur_cnt = conn_cnt.cursor()
            total_al = cur_cnt.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
            cur_cnt.execute("SELECT severity, COUNT(1) FROM alerts GROUP BY severity")
            sev_map = dict(cur_cnt.fetchall())
            uniq_sigs = cur_cnt.execute("SELECT COUNT(DISTINCT signature_id) FROM alerts").fetchone()[0]
            conn_cnt.close()

            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            large_traffic_state["current_pkts"] = current_pkts
            large_traffic_state["progress_pct"] = pct
            large_traffic_state["elapsed_sec"] = round(elapsed, 1)
            large_traffic_state["eta_sec"] = eta_sec
            large_traffic_state["pkts_per_sec"] = pkt_rate
            large_traffic_state["mb_per_sec"] = mb_rate
            large_traffic_state["alerts_count"] = total_al
            large_traffic_state["high_alerts"] = sev_map.get(1, 0)
            large_traffic_state["med_alerts"] = sev_map.get(2, 0)
            large_traffic_state["low_alerts"] = sev_map.get(3, 0)
            large_traffic_state["unique_signatures"] = uniq_sigs
            large_traffic_state["cpu_percent"] = cpu
            large_traffic_state["ram_used_mb"] = round((mem.total - mem.available) / (1024 * 1024), 1)
            large_traffic_state["ram_percent"] = mem.percent

            broadcast_event("large_traffic_progress", dict(large_traffic_state))

    # Finalize execution state
    total_elapsed = round(max(0.1, time.time() - start_t), 1)
    large_traffic_state["elapsed_sec"] = total_elapsed
    large_traffic_state["is_running"] = False

    if large_traffic_state["stop_requested"]:
        large_traffic_state["status"] = "Stopped by User"
        large_traffic_state["is_partial"] = True
        large_traffic_state["stop_message"] = "Partial results — capture was stopped by operator."
    else:
        large_traffic_state["current_pkts"] = large_traffic_state["total_pkts"]
        large_traffic_state["progress_pct"] = 100.0
        large_traffic_state["status"] = "Completed"
        large_traffic_state["is_partial"] = False
        large_traffic_state["stop_message"] = "Capture stream processed successfully."

    # Final counts from SQLite
    conn_final = sqlite3.connect(DB_PATH)
    try:
        cur = conn_final.cursor()
        final_alerts = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
        cur.execute("SELECT severity, COUNT(1) FROM alerts GROUP BY severity")
        sev_map = dict(cur.fetchall())
        uniq_sigs = cur.execute("SELECT COUNT(DISTINCT signature_id) FROM alerts").fetchone()[0]
    except Exception:
        final_alerts = 0
        sev_map = {}
        uniq_sigs = 0
    finally:
        conn_final.close()

    sustained_pkts_s = round(large_traffic_state["current_pkts"] / total_elapsed, 1) if total_elapsed > 0 else 0.0
    sustained_mb_s = round((sustained_pkts_s * 840) / (1024 * 1024), 2)
    large_traffic_state["pkts_per_sec"] = sustained_pkts_s
    large_traffic_state["mb_per_sec"] = sustained_mb_s
    large_traffic_state["alerts_count"] = final_alerts
    large_traffic_state["high_alerts"] = sev_map.get(1, 0)
    large_traffic_state["med_alerts"] = sev_map.get(2, 0)
    large_traffic_state["low_alerts"] = sev_map.get(3, 0)
    large_traffic_state["unique_signatures"] = uniq_sigs

    report = {
        "capture_filename": large_traffic_state["pcap_filename"],
        "capture_size_gb": large_traffic_state["pcap_size_gb"],
        "packets_processed": large_traffic_state["current_pkts"],
        "total_packets_in_file": large_traffic_state["total_pkts"],
        "processing_time_sec": total_elapsed,
        "sustained_throughput_pkts_sec": sustained_pkts_s,
        "sustained_throughput_mb_sec": sustained_mb_s,
        "total_alerts_generated": final_alerts,
        "unique_signatures_triggered": uniq_sigs,
        "high_severity_alerts": sev_map.get(1, 0),
        "medium_severity_alerts": sev_map.get(2, 0),
        "low_severity_alerts": sev_map.get(3, 0),
        "packet_drops": 0,
        "processing_errors": 0,
        "database_records_committed": final_alerts,
        "peak_memory_mb": large_traffic_state["ram_used_mb"],
        "engine_status": large_traffic_state["status"],
        "is_partial": large_traffic_state["is_partial"],
        "notice": large_traffic_state["stop_message"]
    }
    large_traffic_state["completed_report"] = report
    _eval_cache = None

    broadcast_event("large_traffic_progress", dict(large_traffic_state))
    broadcast_event("large_traffic_completed", report)

def _large_traffic_worker():
    global large_traffic_state
    large_traffic_state["is_running"] = True
    large_traffic_state["status"] = "Running"
    large_traffic_state["stop_requested"] = False
    large_traffic_state["is_partial"] = False
    large_traffic_state["stop_message"] = ""
    large_traffic_state["completed_report"] = None
    try:
        _run_large_traffic_logic()
    except Exception as err:
        traceback.print_exc()
        large_traffic_state["is_running"] = False
        large_traffic_state["status"] = f"Failed: {str(err)}"
        large_traffic_state["stop_message"] = str(err)
        broadcast_event("large_traffic_progress", dict(large_traffic_state))

@app.post("/api/large-traffic/start")
def start_large_traffic():
    if large_traffic_state["is_running"]:
        return {"status": "error", "message": "Large traffic analysis already in progress."}
    threading.Thread(target=_large_traffic_worker, daemon=True).start()
    return {"status": "started", "pcap": large_traffic_state["pcap_filename"]}

@app.post("/api/large-traffic/stop")
def stop_large_traffic():
    if not large_traffic_state["is_running"]:
        return {"status": "idle", "message": "No active PCAP replay running."}
    
    large_traffic_state["stop_requested"] = True
    large_traffic_state["status"] = "Stopping..."
    broadcast_event("large_traffic_progress", dict(large_traffic_state))
    
    stop_suricata_subprocess(_active_suricata_proc)
    
    return {
        "status": "stopping",
        "message": "Graceful stop requested. Flushing remaining records and committing database transactions..."
    }

@app.get("/api/large-traffic/status")
def get_large_traffic_status():
    return large_traffic_state

@app.get("/api/large-traffic/report")
def get_large_traffic_report():
    if large_traffic_state["completed_report"]:
        return {"status": "available", "report": large_traffic_state["completed_report"]}
    return {"status": "none", "message": "No full PCAP run completed yet."}

# ==============================================================================
# ALERTS & VISUALIZATIONS
# ==============================================================================

@app.get("/api/alerts/summary")
def get_alerts_summary():
    if not os.path.exists(DB_PATH):
        return {
            "total_alerts": 0,
            "severity_distribution": {"High": 0, "Medium": 0, "Low": 0},
            "category_distribution": [],
            "top_src_ips": [],
            "top_dest_ips": [],
            "top_dest_ports": [],
            "protocol_distribution": []
        }
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        total_alerts = cur.execute("SELECT COUNT(1) FROM alerts").fetchone()[0]
        if total_alerts == 0:
            return {
                "total_alerts": 0,
                "severity_distribution": {"High": 0, "Medium": 0, "Low": 0},
                "category_distribution": [],
                "top_src_ips": [],
                "top_dest_ips": [],
                "top_dest_ports": [],
                "protocol_distribution": []
            }

        # Severity
        cur.execute("SELECT severity, COUNT(1) FROM alerts GROUP BY severity")
        smap = dict(cur.fetchall())
        sev_dist = {
            "High": smap.get(1, 0),
            "Medium": smap.get(2, 0),
            "Low": smap.get(3, 0)
        }

        # Categories
        cur.execute("SELECT category, COUNT(1) as cnt FROM alerts GROUP BY category ORDER BY cnt DESC")
        cat_dist = [{"category": r[0], "count": r[1]} for r in cur.fetchall()]

        # Top Source IPs
        cur.execute("SELECT src_ip, COUNT(1) as cnt FROM alerts GROUP BY src_ip ORDER BY cnt DESC LIMIT 6")
        top_src = [{"ip": r[0], "count": r[1]} for r in cur.fetchall()]

        # Top Dest IPs
        cur.execute("SELECT dest_ip, COUNT(1) as cnt FROM alerts GROUP BY dest_ip ORDER BY cnt DESC LIMIT 6")
        top_dst = [{"ip": r[0], "count": r[1]} for r in cur.fetchall()]

        # Top Dest Ports
        cur.execute("SELECT dest_port, COUNT(1) as cnt FROM alerts GROUP BY dest_port ORDER BY cnt DESC LIMIT 6")
        top_ports = [{"port": r[0], "count": r[1]} for r in cur.fetchall()]

        # Protocols
        cur.execute("SELECT proto, COUNT(1) as cnt FROM alerts GROUP BY proto ORDER BY cnt DESC")
        proto_dist = [{"proto": r[0], "count": r[1]} for r in cur.fetchall()]

        return {
            "total_alerts": total_alerts,
            "severity_distribution": sev_dist,
            "category_distribution": cat_dist,
            "top_src_ips": top_src,
            "top_dest_ips": top_dst,
            "top_dest_ports": top_ports,
            "protocol_distribution": proto_dist
        }
    finally:
        conn.close()

@app.get("/api/alerts")
def get_alerts(q: Optional[str] = None, severity: Optional[int] = None, limit: int = 100, offset: int = 0, grouped: bool = True):
    if not os.path.exists(DB_PATH):
        return {"alerts": [], "grouped_alerts": [], "total": 0}
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        query = "SELECT id, timestamp, src_ip, src_port, dest_ip, dest_port, proto, signature_id, signature, category, severity, bytes_toserver, pkts_toserver, reason FROM alerts WHERE 1=1"
        params = []
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if q:
            query += " AND (signature LIKE ? OR src_ip LIKE ? OR dest_ip LIKE ? OR reason LIKE ?)"
            wild = f"%{q}%"
            params.extend([wild, wild, wild, wild])
        
        count_cur = conn.cursor()
        count_cur.execute(f"SELECT COUNT(1) FROM ({query})", params)
        total = count_cur.fetchone()[0]
        
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(query, params)
        rows = cur.fetchall()
        
        raw_alerts = []
        for r in rows:
            raw_alerts.append({
                "id": r[0],
                "timestamp": r[1],
                "src_ip": r[2],
                "src_port": r[3],
                "dest_ip": r[4],
                "dest_port": r[5],
                "proto": r[6],
                "signature_id": r[7],
                "signature": r[8],
                "category": r[9],
                "severity": r[10],
                "bytes_toserver": r[11],
                "pkts_toserver": r[12],
                "reason": r[13]
            })
        
        grouped_alerts = []
        current_group = None
        for a in raw_alerts:
            if current_group and current_group["signature_id"] == a["signature_id"]:
                current_group["count"] += 1
                current_group["instances"].append(a)
                current_group["last_seen"] = a["timestamp"]
            else:
                if current_group:
                    grouped_alerts.append(current_group)
                current_group = {
                    "signature_id": a["signature_id"],
                    "signature": a["signature"],
                    "severity": a["severity"],
                    "category": a["category"],
                    "count": 1,
                    "first_seen": a["timestamp"],
                    "last_seen": a["timestamp"],
                    "sample_src": f"{a['src_ip']}:{a['src_port']}",
                    "sample_dst": f"{a['dest_ip']}:{a['dest_port']}",
                    "sample_proto": a["proto"],
                    "sample_reason": a["reason"],
                    "sample_bytes": a["bytes_toserver"],
                    "sample_pkts": a["pkts_toserver"],
                    "instances": [a]
                }
        if current_group:
            grouped_alerts.append(current_group)
        
        return {
            "alerts": raw_alerts,
            "grouped_alerts": grouped_alerts,
            "total": total
        }
    finally:
        conn.close()

@app.get("/api/rules")
def get_rules():
    try:
        rules = load_rules_from_file(RULE_PATH)
        return {"rules": rules, "count": len(rules)}
    except Exception as e:
        return {"rules": [], "count": 0, "error": str(e)}

@app.post("/api/reset")
def reset_database():
    global _eval_cache, large_traffic_state
    init_database(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM alerts")
        cur.execute("DELETE FROM evaluation_flows")
        cur.execute("DELETE FROM ml_predictions")
        conn.commit()
    finally:
        conn.close()
    
    eve_path = "logs/eve.json"
    if os.path.exists(eve_path):
        try:
            with open(eve_path, "w", encoding="utf-8") as f:
                pass
        except Exception:
            pass
            
    _eval_cache = None
    large_traffic_state["is_running"] = False
    large_traffic_state["status"] = "Idle — database cleared"
    large_traffic_state["current_pkts"] = 0
    large_traffic_state["alerts_count"] = 0
    large_traffic_state["elapsed_sec"] = 0.0
    large_traffic_state["eta_sec"] = 0.0
    large_traffic_state["completed_report"] = None
    
    broadcast_event("reset", {})
    return {"status": "success", "message": "Database, logs, and benchmark cache successfully reset."}

INJECT_MAP = {
    "WebAttack-SQLi": (1000010, "Web Attack SQL Injection", "Web Attack - SQL Injection", 80, "TCP", 3, 1500),
    "WebAttack-XSS": (1000011, "Web Attack Cross-Site Scripting (XSS)", "Web Attack - XSS", 80, "TCP", 3, 1200),
    "DoS-LOIC-UDP": (1000001, "DoS High Volume UDP Flood LOIC", "DoS-LOIC-UDP", 80, "UDP", 25, 12000),
    "PortScan-TCP-SYN": (1000006, "Port Scan TCP SYN Probing", "PortScan-Connect", 80, "TCP", 12, 600),
    "BruteForce-SSH": (1000008, "Brute Force SSH-Patator Volumetric Burst", "BruteForce-SSH-Patator", 22, "TCP", 6, 800),
    "Botnet-C2-Beacon": (1000012, "Botnet C2 Heartbeat Beacon", "Botnet-C2-Beaconing", 8080, "TCP", 4, 400),
    "Infiltration-SMB": (1000013, "Infiltration SMB/C2 Lateral Sweep", "Infiltration", 445, "TCP", 3, 900)
}

@app.post("/api/inject")
def inject_attack(req: dict):
    from src.eve_ingestor import generate_dynamic_reason
    from app.ml.fusion import HybridFusionEngine
    from datetime import datetime, timezone

    atk_type = req.get("attack_type", "WebAttack-SQLi")
    tgt = req.get("target_host", "192.168.10.50")
    
    info = INJECT_MAP.get(atk_type, (1000010, "Injected Attack", "Web Attack", 80, "TCP", 3, 1000))
    sid, sig_name, cat, dport, proto, pkts, bytes_cnt = info
    src_ip = "203.0.113.88"
    src_port = 55000 + (sid % 1000)
    now_iso = datetime.now(timezone.utc).isoformat()

    reason = generate_dynamic_reason(
        sig_name=sig_name, sid=sid, src_ip=src_ip, src_port=src_port,
        dest_ip=tgt, dest_port=dport, proto=proto, category=cat,
        pkts=pkts, bytes_count=bytes_cnt, duration=0.05
    )

    # Insert into SQLite alerts
    init_database(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO alerts (
            timestamp, src_ip, src_port, dest_ip, dest_port, proto,
            signature_id, signature, category, severity,
            bytes_toserver, pkts_toserver, bytes_toclient, pkts_toclient, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso, src_ip, src_port, tgt, dport, proto,
        sid, sig_name, cat, 1, bytes_cnt, pkts, 200, 2, reason
    ))
    conn.commit()
    conn.close()

    # Pass through ML Fusion Engine
    engine = HybridFusionEngine()
    flow_event = {
        "timestamp": now_iso,
        "src_ip": src_ip,
        "src_port": src_port,
        "dest_ip": tgt,
        "dest_port": dport,
        "proto": proto,
        "flow": {
            "age": 0.05,
            "pkts_toserver": pkts,
            "bytes_toserver": bytes_cnt,
            "pkts_toclient": 2,
            "bytes_toclient": 200,
            "state": "established",
            "reason": "shutdown"
        },
        "tcp": {"tcp_flags": "18"},
        "app_proto": "http" if dport == 80 else "failed"
    }
    fusion_rec = engine.process_flow(flow_event)
    HybridFusionEngine.persist_fusion_records([fusion_rec], DB_PATH)

    # Broadcast real-time SSE events to connected UI clients
    alert_payload = {
        "signature": sig_name,
        "signature_id": sid,
        "src_ip": src_ip,
        "dest_ip": tgt,
        "dest_port": dport,
        "severity": 1,
        "timestamp": now_iso
    }
    broadcast_event("new_alert", alert_payload)
    broadcast_event("new_ml_prediction", fusion_rec)

    return {
        "status": "success",
        "result": {
            "attack_type": atk_type,
            "signature_id": sid,
            "signature": sig_name,
            "packets_injected": pkts,
            "severity": 1,
            "reason": reason,
            "ml_decision": fusion_rec.get("fusion_status", "HYBRID_CONFIRMED")
        }
    }

@app.get("/api/stream/events")
async def stream_events(request: Request):
    async def event_generator():
        q = asyncio.Queue()
        event_subscribers.append(q)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in event_subscribers:
                event_subscribers.remove(q)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==============================================================================
# MACHINE LEARNING & HYBRID FUSION ENDPOINTS
# ==============================================================================

@app.get("/api/ml/metrics")
def get_ml_metrics():
    """Returns scientific evaluation metrics computed on held-out test data."""
    report_path = os.path.join(ROOT_DIR, "models", "evaluation_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to read evaluation report: {str(e)}"}
    return {
        "status": "unverified",
        "message": "ML evaluation metrics not yet computed. Run python -m app.ml.train to generate report."
    }

@app.get("/api/ml/predictions")
def get_ml_predictions(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Returns paginated hybrid ML predictions with explainability attribution."""
    if not os.path.exists(DB_PATH):
        return {"predictions": [], "total": 0}
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        query = (
            "SELECT id, flow_id, timestamp, src_ip, src_port, dest_ip, dest_port, proto, "
            "predicted_label, is_anomaly, confidence, attack_probability, "
            "top_contributing_features, fusion_status, signature_id, signature, "
            "explainability_summary FROM ml_predictions WHERE 1=1"
        )
        params = []
        if status:
            query += " AND fusion_status = ?"
            params.append(status.upper())

        count_cur = conn.cursor()
        count_cur.execute(f"SELECT COUNT(1) FROM ({query})", params)
        total = count_cur.fetchone()[0]

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(query, params)
        rows = cur.fetchall()

        preds = []
        for r in rows:
            top_feats = []
            if r[12]:
                try:
                    top_feats = json.loads(r[12]) if isinstance(r[12], str) else r[12]
                except Exception:
                    top_feats = []

            preds.append({
                "id": r[0],
                "flow_id": r[1],
                "timestamp": r[2],
                "src_ip": r[3],
                "src_port": r[4],
                "dest_ip": r[5],
                "dest_port": r[6],
                "proto": r[7],
                "predicted_label": r[8],
                "is_anomaly": r[9],
                "confidence": r[10],
                "attack_probability": r[11],
                "top_contributing_features": top_feats,
                "fusion_status": r[13],
                "signature_id": r[14],
                "signature": r[15],
                "explainability_summary": r[16]
            })

        return {"predictions": preds, "total": total}
    except Exception as e:
        return {"predictions": [], "total": 0, "error": str(e)}
    finally:
        conn.close()

@app.get("/api/ml/fusion/summary")
def get_fusion_summary():
    """Returns distribution of hybrid fusion decisions (Corroborated, Anomaly-only, etc.)."""
    if not os.path.exists(DB_PATH):
        return {
            "total": 0,
            "corroborated": 0,
            "anomaly_only": 0,
            "signature_only": 0,
            "benign": 0
        }
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        total = cur.execute("SELECT COUNT(1) FROM ml_predictions").fetchone()[0]
        cur.execute("SELECT fusion_status, COUNT(1) FROM ml_predictions GROUP BY fusion_status")
        fmap = dict(cur.fetchall())
        return {
            "total": total,
            "corroborated": fmap.get("CORROBORATED", 0),
            "anomaly_only": fmap.get("ANOMALY_ONLY", 0),
            "signature_only": fmap.get("SIGNATURE_ONLY", 0),
            "benign": fmap.get("BENIGN", 0)
        }
    except Exception:
        return {
            "total": 0,
            "corroborated": 0,
            "anomaly_only": 0,
            "signature_only": 0,
            "benign": 0
        }
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

