"""
CLI script to extract ML features from Suricata EVE flow records.
Reads logs/eve.json (or specified input) and outputs extracted features to CSV.
"""

import os
import sys
import argparse

# Ensure repo root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ml.features import parse_eve_log_file, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Extract ML features from Suricata EVE flow logs")
    parser.add_argument("--input", "-i", type=str, default="logs/eve.json", help="Path to input eve.json")
    parser.add_argument("--output", "-o", type=str, default="data/extracted_features.csv", help="Path to output CSV")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Maximum flow records to extract")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"[*] Parsing EVE flow records from: {args.input}")
    df = parse_eve_log_file(args.input, max_records=args.limit)
    print(f"[+] Successfully extracted {len(df)} flow records ({len(FEATURE_NAMES)} features each).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[+] Extracted feature dataset saved to: {args.output}")


if __name__ == "__main__":
    main()

