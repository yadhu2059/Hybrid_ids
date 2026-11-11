"""
Unit tests for app.ml.features
Tests single-record and batch feature extraction, boundary conditions,
division-by-zero guards, TCP flag decoding, and schema conformity.
"""

import math
import pytest
import pandas as pd
import numpy as np

from app.ml.features import (
    FEATURE_NAMES,
    METADATA_NAMES,
    extract_features_from_flow,
    extract_flow_metadata,
    extract_features_batch,
    _extract_tcp_flags,
    _parse_duration
)


def test_feature_names_count_and_types():
    assert len(FEATURE_NAMES) == 25
    assert "duration" in FEATURE_NAMES
    assert "pkts_per_sec" in FEATURE_NAMES
    assert "bytes_per_sec" in FEATURE_NAMES
    assert "bytes_ratio" in FEATURE_NAMES
    assert "pkts_ratio" in FEATURE_NAMES
    assert "dest_port" in FEATURE_NAMES
    assert "tcp_flag_syn" in FEATURE_NAMES


def test_extract_features_single_typical_flow():
    sample_flow = {
        "timestamp": "2017-07-06T13:20:05.000000+00:00",
        "flow_id": 987654321,
        "event_type": "flow",
        "src_ip": "172.16.0.1",
        "src_port": 49152,
        "dest_ip": "192.168.10.50",
        "dest_port": 80,
        "proto": "TCP",
        "app_proto": "http",
        "flow": {
            "pkts_toserver": 10,
            "pkts_toclient": 5,
            "bytes_toserver": 1500,
            "bytes_toclient": 3000,
            "start": "2017-07-06T13:20:00.000000+00:00",
            "end": "2017-07-06T13:20:02.500000+00:00",
            "age": 2.5,
            "state": "established",
            "reason": "shutdown"
        },
        "tcp": {
            "syn": True,
            "fin": True,
            "rst": False,
            "psh": True,
            "ack": True
        }
    }

    features = extract_features_from_flow(sample_flow)

    assert set(features.keys()) == set(FEATURE_NAMES)
    assert features["duration"] == pytest.approx(2.5, rel=1e-3)
    assert features["pkts_toserver"] == 10.0
    assert features["pkts_toclient"] == 5.0
    assert features["total_pkts"] == 15.0
    assert features["bytes_toserver"] == 1500.0
    assert features["bytes_toclient"] == 3000.0
    assert features["total_bytes"] == 4500.0
    assert features["bytes_per_sec"] == pytest.approx(4500.0 / 2.5, rel=1e-3)
    assert features["pkts_per_sec"] == pytest.approx(15.0 / 2.5, rel=1e-3)
    assert features["bytes_ratio"] == pytest.approx(1500.0 / 3000.0, rel=1e-3)
    assert features["pkts_ratio"] == pytest.approx(10.0 / 5.0, rel=1e-3)
    assert features["avg_pkt_size"] == pytest.approx(4500.0 / 15.0, rel=1e-3)
    assert features["dest_port"] == 80.0
    assert features["dest_port_is_privileged"] == 1.0
    assert features["proto_num"] == 6.0  # TCP
    assert features["app_proto_code"] == 1.0  # http
    assert features["flow_state_code"] == 2.0  # established
    assert features["flow_reason_code"] == 2.0  # shutdown
    assert features["tcp_flag_syn"] == 1.0
    assert features["tcp_flag_fin"] == 1.0
    assert features["tcp_flag_rst"] == 0.0
    assert features["tcp_flag_ack"] == 1.0


def test_extract_features_json_string_input():
    raw_json = '{"src_ip": "10.0.0.1", "dest_port": 443, "proto": "TCP", "flow": {"pkts_toserver": 4, "bytes_toserver": 200, "age": 1.0}}'
    features = extract_features_from_flow(raw_json)
    assert features["dest_port"] == 443.0
    assert features["dest_port_is_privileged"] == 1.0
    assert features["pkts_toserver"] == 4.0
    assert features["bytes_toserver"] == 200.0
    assert features["duration"] == 1.0


def test_extract_features_empty_and_malformed():
    for malformed in [{}, "", "null", "{invalid_json", None]:
        res = extract_features_from_flow(malformed)
        assert isinstance(res, dict)
        assert set(res.keys()) == set(FEATURE_NAMES)
        for k, v in res.items():
            assert not math.isnan(v), f"Feature {k} was NaN on malformed input"
            assert not math.isinf(v), f"Feature {k} was Inf on malformed input"


def test_zero_duration_and_rates_guard():
    zero_flow = {
        "dest_port": 8080,
        "proto": "TCP",
        "flow": {
            "pkts_toserver": 100,
            "bytes_toserver": 5000,
            "pkts_toclient": 0,
            "bytes_toclient": 0,
            "age": 0.0
        }
    }
    features = extract_features_from_flow(zero_flow)
    assert features["duration"] == pytest.approx(0.0001)
    assert features["bytes_per_sec"] > 0
    assert not math.isinf(features["bytes_per_sec"])
    assert not math.isnan(features["bytes_per_sec"])
    assert features["bytes_ratio"] == pytest.approx(5000.0 / 1.0)
    assert features["pkts_ratio"] == pytest.approx(100.0 / 1.0)


def test_tcp_flags_hex_bitmask_decoding():
    # 0x1b = FIN (1) | SYN (2) | PSH (8) | ACK (16)
    tcp_hex = {"tcp_flags": "1b"}
    flags = _extract_tcp_flags(tcp_hex)
    assert flags["tcp_flag_fin"] == 1
    assert flags["tcp_flag_syn"] == 1
    assert flags["tcp_flag_rst"] == 0
    assert flags["tcp_flag_psh"] == 1
    assert flags["tcp_flag_ack"] == 1

    # 0x04 = RST (4)
    tcp_rst = {"tcp_flags": "04"}
    flags_rst = _extract_tcp_flags(tcp_rst)
    assert flags_rst["tcp_flag_rst"] == 1
    assert flags_rst["tcp_flag_syn"] == 0


def test_duration_parsing_from_iso_timestamps():
    flow_obj = {
        "start": "2017-07-06T13:20:00.000000Z",
        "end": "2017-07-06T13:20:04.500000Z"
    }
    dur = _parse_duration(flow_obj, None)
    assert dur == pytest.approx(4.5, rel=1e-3)


def test_udp_and_icmp_protocols():
    udp_flow = {
        "dest_port": 53,
        "proto": "UDP",
        "app_proto": "dns",
        "flow": {
            "pkts_toserver": 1,
            "pkts_toclient": 1,
            "bytes_toserver": 45,
            "bytes_toclient": 120,
            "age": 0.05
        }
    }
    res_udp = extract_features_from_flow(udp_flow)
    assert res_udp["proto_num"] == 17.0
    assert res_udp["app_proto_code"] == 5.0  # dns
    assert res_udp["tcp_flag_syn"] == 0.0

    icmp_flow = {
        "dest_port": 0,
        "proto": "ICMP",
        "flow": {"pkts_toserver": 4, "bytes_toserver": 256, "age": 1.0}
    }
    res_icmp = extract_features_from_flow(icmp_flow)
    assert res_icmp["proto_num"] == 1.0


def test_extract_flow_metadata():
    evt = {
        "flow_id": 11223344,
        "timestamp": "2017-07-06T13:20:00Z",
        "src_ip": "10.0.0.5",
        "src_port": 54321,
        "dest_ip": "10.0.0.1",
        "dest_port": 443,
        "proto": "tcp"
    }
    meta = extract_flow_metadata(evt)
    assert meta["flow_id"] == "11223344"
    assert meta["src_ip"] == "10.0.0.5"
    assert meta["src_port"] == 54321
    assert meta["dest_ip"] == "10.0.0.1"
    assert meta["dest_port"] == 443
    assert meta["proto"] == "TCP"


def test_extract_features_batch():
    records = [
        {
            "flow_id": 1,
            "dest_port": 80,
            "proto": "TCP",
            "flow": {"pkts_toserver": 5, "bytes_toserver": 500, "age": 1.0}
        },
        {
            "flow_id": 2,
            "dest_port": 22,
            "proto": "TCP",
            "flow": {"pkts_toserver": 20, "bytes_toserver": 2000, "age": 2.0}
        }
    ]

    df = extract_features_batch(records, include_metadata=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == FEATURE_NAMES
    assert df["dest_port"].tolist() == [80.0, 22.0]

    # Test with metadata
    df_meta = extract_features_batch(records, include_metadata=True)
    assert len(df_meta) == 2
    expected_cols = list(dict.fromkeys(METADATA_NAMES + FEATURE_NAMES))
    assert list(df_meta.columns) == expected_cols
    assert df_meta["flow_id"].tolist() == ["1", "2"]
    for f in FEATURE_NAMES:
        assert f in df_meta.columns

    # Test from DataFrame input
    df_input = pd.DataFrame(records)
    df_from_df = extract_features_batch(df_input)
    assert len(df_from_df) == 2
    assert list(df_from_df.columns) == FEATURE_NAMES


def test_batch_guarantees_no_nan_or_inf():
    dirty_records = [
        {"dest_port": "invalid", "flow": {"age": -5.0, "bytes_toserver": None}},
        {"dest_port": None, "flow": None},
        {}
    ]
    df = extract_features_batch(dirty_records)
    assert len(df) == 3
    assert not df.isna().any().any()
    assert not np.isinf(df.to_numpy()).any()
