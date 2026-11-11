"""
Shared Feature Extraction Engine for Hybrid Intrusion Detection System
Derives ML features strictly from Suricata EVE flow records.
Used by both offline training pipelines and live real-time inference to guarantee 100% parity.
"""

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

# Canonical feature names strictly ordered
FEATURE_NAMES: List[str] = [
    "duration",
    "pkts_toserver",
    "pkts_toclient",
    "total_pkts",
    "bytes_toserver",
    "bytes_toclient",
    "total_bytes",
    "bytes_per_sec",
    "pkts_per_sec",
    "bytes_ratio",
    "pkts_ratio",
    "avg_pkt_size",
    "avg_pkt_size_toserver",
    "avg_pkt_size_toclient",
    "dest_port",
    "dest_port_is_privileged",
    "proto_num",
    "app_proto_code",
    "flow_state_code",
    "flow_reason_code",
    "tcp_flag_syn",
    "tcp_flag_fin",
    "tcp_flag_rst",
    "tcp_flag_psh",
    "tcp_flag_ack"
]

METADATA_NAMES: List[str] = [
    "flow_id",
    "timestamp",
    "src_ip",
    "src_port",
    "dest_ip",
    "dest_port",
    "proto"
]

PROTO_MAP: Dict[str, int] = {
    "HOPOPT": 0,
    "ICMP": 1,
    "IGMP": 2,
    "GGP": 3,
    "IPV4": 4,
    "ST": 5,
    "TCP": 6,
    "CBT": 7,
    "EGP": 8,
    "IGP": 9,
    "UDP": 17,
    "IPV6": 41,
    "GRE": 47,
    "ESP": 50,
    "AH": 51
}

APP_PROTO_MAP: Dict[str, int] = {
    "unknown": 0,
    "failed": 0,
    "http": 1,
    "tls": 2,
    "ssh": 3,
    "ftp": 4,
    "dns": 5,
    "smb": 6,
    "dhcp": 7,
    "dcerpc": 8,
    "ntp": 9,
    "snmp": 10,
    "smtp": 11,
    "imap": 12,
    "rdp": 13,
    "sip": 14,
    "nfs": 15
}

FLOW_STATE_MAP: Dict[str, int] = {
    "unknown": 0,
    "new": 1,
    "established": 2,
    "closed": 3,
    "bypassed": 4
}

FLOW_REASON_MAP: Dict[str, int] = {
    "unknown": 0,
    "timeout": 1,
    "shutdown": 2,
    "forced": 3
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Converts a value to float safely, handling None, NaN, and Inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Converts a value to int safely, handling strings and non-numeric inputs."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _parse_duration(flow_obj: Dict[str, Any], raw_age: Any) -> float:
    """Calculates flow duration in seconds with microsecond resolution."""
    age = _safe_float(raw_age, -1.0)
    if age > 0.0:
        return max(age, 0.0001)

    start_str = flow_obj.get("start")
    end_str = flow_obj.get("end")
    if start_str and end_str:
        try:
            # Strip trailing Z or +00:00 for robust parsing
            s_clean = str(start_str).replace("Z", "+00:00")
            e_clean = str(end_str).replace("Z", "+00:00")
            dt_start = datetime.fromisoformat(s_clean)
            dt_end = datetime.fromisoformat(e_clean)
            diff = (dt_end - dt_start).total_seconds()
            return max(diff, 0.0001)
        except Exception:
            pass

    return 0.0001


def _extract_tcp_flags(tcp_obj: Dict[str, Any]) -> Dict[str, int]:
    """
    Extracts TCP flags supporting boolean flags, integer counts,
    or hexadecimal flag bitmasks (Suricata tcp_flags format).
    """
    flags = {
        "tcp_flag_syn": 0,
        "tcp_flag_fin": 0,
        "tcp_flag_rst": 0,
        "tcp_flag_psh": 0,
        "tcp_flag_ack": 0
    }
    if not tcp_obj or not isinstance(tcp_obj, dict):
        return flags

    # 1. Direct boolean / count keys
    for k, flag_name in [
        ("syn", "tcp_flag_syn"),
        ("fin", "tcp_flag_fin"),
        ("rst", "tcp_flag_rst"),
        ("psh", "tcp_flag_psh"),
        ("ack", "tcp_flag_ack")
    ]:
        val = tcp_obj.get(k)
        if isinstance(val, bool):
            flags[flag_name] = 1 if val else 0
        elif isinstance(val, (int, float)):
            flags[flag_name] = int(val)

    # 2. Check for hex string in tcp_flags (e.g., '1b' -> 0x1b)
    hex_flags = tcp_obj.get("tcp_flags")
    if hex_flags is not None and flags["tcp_flag_syn"] == 0 and flags["tcp_flag_fin"] == 0:
        try:
            val_int = int(str(hex_flags), 16) if isinstance(hex_flags, str) else int(hex_flags)
            flags["tcp_flag_fin"] = 1 if (val_int & 0x01) else 0
            flags["tcp_flag_syn"] = 1 if (val_int & 0x02) else 0
            flags["tcp_flag_rst"] = 1 if (val_int & 0x04) else 0
            flags["tcp_flag_psh"] = 1 if (val_int & 0x08) else 0
            flags["tcp_flag_ack"] = 1 if (val_int & 0x10) else 0
        except Exception:
            pass

    return flags


def extract_features_from_flow(event: Union[Dict[str, Any], str]) -> Dict[str, float]:
    """
    Extracts the canonical numeric feature vector from a single Suricata EVE flow event.

    Args:
        event: Dict representing an EVE JSON event, or a raw JSON string.

    Returns:
        Dict[str, float] containing exactly the keys defined in FEATURE_NAMES.
    """
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except Exception:
            event = {}

    if not isinstance(event, dict):
        event = {}

    flow_obj = event.get("flow", {}) if isinstance(event.get("flow"), dict) else {}
    tcp_obj = event.get("tcp", {}) if isinstance(event.get("tcp"), dict) else {}

    # 1. Packet & Byte counters
    pkts_toserver = _safe_float(flow_obj.get("pkts_toserver", event.get("pkts_toserver", 0)))
    pkts_toclient = _safe_float(flow_obj.get("pkts_toclient", event.get("pkts_toclient", 0)))
    total_pkts = pkts_toserver + pkts_toclient

    bytes_toserver = _safe_float(flow_obj.get("bytes_toserver", event.get("bytes_toserver", 0)))
    bytes_toclient = _safe_float(flow_obj.get("bytes_toclient", event.get("bytes_toclient", 0)))
    total_bytes = bytes_toserver + bytes_toclient

    # 2. Duration & Rates
    raw_age = flow_obj.get("age", event.get("duration", event.get("age", 0.0)))
    duration = _parse_duration(flow_obj, raw_age)

    bytes_per_sec = total_bytes / duration
    pkts_per_sec = total_pkts / duration

    # 3. Ratios & Averages (div-by-zero guarded)
    bytes_ratio = bytes_toserver / max(bytes_toclient, 1.0)
    pkts_ratio = pkts_toserver / max(pkts_toclient, 1.0)

    avg_pkt_size = total_bytes / max(total_pkts, 1.0)
    avg_pkt_size_toserver = bytes_toserver / max(pkts_toserver, 1.0)
    avg_pkt_size_toclient = bytes_toclient / max(pkts_toclient, 1.0)

    # 4. Port & Protocol Metadata
    dest_port = _safe_int(event.get("dest_port", 0))
    dest_port_is_privileged = 1.0 if (0 < dest_port < 1024) else 0.0

    raw_proto = str(event.get("proto", "TCP")).strip().upper()
    proto_num = float(PROTO_MAP.get(raw_proto, _safe_int(raw_proto, 6)))

    raw_app_proto = str(event.get("app_proto", "unknown")).strip().lower()
    app_proto_code = float(APP_PROTO_MAP.get(raw_app_proto, 0))

    # 5. Flow State & Reason
    raw_state = str(flow_obj.get("state", event.get("state", "unknown"))).strip().lower()
    flow_state_code = float(FLOW_STATE_MAP.get(raw_state, 0))

    raw_reason = str(flow_obj.get("reason", event.get("reason", "unknown"))).strip().lower()
    flow_reason_code = float(FLOW_REASON_MAP.get(raw_reason, 0))

    # 6. TCP Flags
    flags = _extract_tcp_flags(tcp_obj)

    feature_dict = {
        "duration": float(duration),
        "pkts_toserver": float(pkts_toserver),
        "pkts_toclient": float(pkts_toclient),
        "total_pkts": float(total_pkts),
        "bytes_toserver": float(bytes_toserver),
        "bytes_toclient": float(bytes_toclient),
        "total_bytes": float(total_bytes),
        "bytes_per_sec": float(bytes_per_sec),
        "pkts_per_sec": float(pkts_per_sec),
        "bytes_ratio": float(bytes_ratio),
        "pkts_ratio": float(pkts_ratio),
        "avg_pkt_size": float(avg_pkt_size),
        "avg_pkt_size_toserver": float(avg_pkt_size_toserver),
        "avg_pkt_size_toclient": float(avg_pkt_size_toclient),
        "dest_port": float(dest_port),
        "dest_port_is_privileged": float(dest_port_is_privileged),
        "proto_num": float(proto_num),
        "app_proto_code": float(app_proto_code),
        "flow_state_code": float(flow_state_code),
        "flow_reason_code": float(flow_reason_code),
        "tcp_flag_syn": float(flags["tcp_flag_syn"]),
        "tcp_flag_fin": float(flags["tcp_flag_fin"]),
        "tcp_flag_rst": float(flags["tcp_flag_rst"]),
        "tcp_flag_psh": float(flags["tcp_flag_psh"]),
        "tcp_flag_ack": float(flags["tcp_flag_ack"])
    }

    # Verify no NaN or Inf in output
    for k, v in feature_dict.items():
        if math.isnan(v) or math.isinf(v):
            feature_dict[k] = 0.0

    return feature_dict


def extract_flow_metadata(event: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Extracts session identifying 5-tuple and timestamp metadata."""
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except Exception:
            event = {}

    if not isinstance(event, dict):
        event = {}

    flow_obj = event.get("flow", {}) if isinstance(event.get("flow"), dict) else {}
    ts = (
        flow_obj.get("start")
        or event.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )

    return {
        "flow_id": str(event.get("flow_id", f"FLOW_{event.get('src_ip', '0')}_{event.get('src_port', 0)}")),
        "timestamp": str(ts),
        "src_ip": str(event.get("src_ip", "0.0.0.0")),
        "src_port": _safe_int(event.get("src_port", 0)),
        "dest_ip": str(event.get("dest_ip", "0.0.0.0")),
        "dest_port": _safe_int(event.get("dest_port", 0)),
        "proto": str(event.get("proto", "TCP")).upper()
    }


def extract_features_batch(
    records: Union[List[Dict[str, Any]], pd.DataFrame],
    include_metadata: bool = False
) -> pd.DataFrame:
    """
    Extracts features for a batch of flow records into a pandas DataFrame.

    Args:
        records: List of flow dictionaries or a DataFrame of flow events.
        include_metadata: If True, prepends METADATA_NAMES columns.

    Returns:
        pd.DataFrame with FEATURE_NAMES columns (and optionally METADATA_NAMES).
    """
    if isinstance(records, pd.DataFrame):
        records = records.to_dict(orient="records")

    rows: List[Dict[str, Any]] = []
    for r in records:
        features = extract_features_from_flow(r)
        if include_metadata:
            meta = extract_flow_metadata(r)
            combined = {**meta, **features}
            rows.append(combined)
        else:
            rows.append(features)

    raw_cols = (METADATA_NAMES + FEATURE_NAMES) if include_metadata else FEATURE_NAMES
    # Deduplicate while preserving order (dest_port exists in both metadata and features)
    seen = set()
    cols: List[str] = []
    for c in raw_cols:
        if c not in seen:
            seen.add(c)
            cols.append(c)

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    # Ensure all columns exist and ordered properly
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[cols]

    # Convert feature columns to float64, ensuring zero NaN/Inf
    for f in FEATURE_NAMES:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
        df[f] = df[f].replace([np.inf, -np.inf], 0.0)

    return df


def parse_eve_log_file(eve_path: str, max_records: int = None) -> pd.DataFrame:
    """
    Parses Suricata eve.json, filters event_type == 'flow',
    and extracts batch features with metadata.
    """
    records = []
    with open(eve_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                if evt.get("event_type") == "flow":
                    records.append(evt)
                    if max_records and len(records) >= max_records:
                        break
            except Exception:
                continue

    return extract_features_batch(records, include_metadata=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Suricata EVE Flow Feature Extractor")
    parser.add_argument("--input", type=str, default="logs/eve.json", help="Path to eve.json")
    parser.add_argument("--output", type=str, default="data/extracted_features.csv", help="Path to output CSV")
    parser.add_argument("--limit", type=int, default=None, help="Max flow records to process")
    args = parser.parse_args()

    print(f"[*] Extracting features from {args.input}...")
    df_features = parse_eve_log_file(args.input, max_records=args.limit)
    print(f"[+] Extracted {len(df_features)} flows with {len(FEATURE_NAMES)} features.")
    if len(df_features) > 0:
        df_features.to_csv(args.output, index=False)
        print(f"[+] Saved extracted features to {args.output}")
