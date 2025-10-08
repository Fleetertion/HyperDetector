#!/usr/bin/env python3
"""
generate_splits_yaml.py
-----------------------
Scan the original APT-2021 CSV and generate a splits.yaml that lists
train_benign, test_benign and test_malicious batch IDs.

* Each “batch” is a fixed-length time window (default 5 minutes).
* A batch is **benign** if all rows have Label == "NormalTraffic";
  otherwise it is **malicious**.
* Benign batches are shuffled and split into train / test with the
  ratio given by --train_ratio (default 0.8).
* The YAML format matches the training pipeline shown earlier.

Dependencies:
    pip install pandas pyyaml
"""

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to APT-2021 CSV file")
    ap.add_argument(
        "--out",
        default="splits.yaml",
        help="Destination YAML file (default: splits.yaml)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=5,
        help="Batch window size in minutes (must match graph split)",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of benign batches sent to the training split",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for shuffling benign batch IDs",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1.  Load CSV and sort by timestamp
    # ------------------------------------------------------------------ #
    df = pd.read_csv(args.csv, parse_dates=["Timestamp"])
    df.sort_values("Timestamp", inplace=True)

    # ------------------------------------------------------------------ #
    # 2.  Assign each row to a batch_id (fixed size time window)
    # ------------------------------------------------------------------ #
    start_time = df["Timestamp"].min()
    window_seconds = args.window * 60
    df["batch_id"] = (
        (df["Timestamp"] - start_time).dt.total_seconds() // window_seconds
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 3.  Decide whether each batch is benign or malicious
    # ------------------------------------------------------------------ #
    def is_benign(label_series: pd.Series) -> bool:
        """Return True if every row in the batch is NormalTraffic."""
        return (label_series == "NormalTraffic").all()

    batch_ok = df.groupby("batch_id")["Label"].apply(is_benign)
    benign_ids = batch_ok[batch_ok].index.tolist()
    malicious_ids = batch_ok[~batch_ok].index.tolist()

    # ------------------------------------------------------------------ #
    # 4.  Split benign batches into train / test
    # ------------------------------------------------------------------ #
    random.seed(args.seed)
    random.shuffle(benign_ids)

    cut = math.ceil(len(benign_ids) * args.train_ratio)
    train_benign = sorted(benign_ids[:cut])
    test_benign = sorted(benign_ids[cut:])
    test_malicious = sorted(malicious_ids)

    # ------------------------------------------------------------------ #
    # 5.  Dump YAML
    # ------------------------------------------------------------------ #
    splits = {
        "apt2021": {
            "train_benign": train_benign,
            "test_benign": test_benign,
            "test_malicious": test_malicious,
        }
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        yaml.safe_dump(splits, fp, sort_keys=False)

    # ------------------------------------------------------------------ #
    # 6.  Console summary
    # ------------------------------------------------------------------ #
    print(f"[✓] splits.yaml written to {args.out}")
    print(f"    Benign batches:    {len(benign_ids)} "
          f"(train {len(train_benign)}, test {len(test_benign)})")
    print(f"    Malicious batches: {len(test_malicious)}")


if __name__ == "__main__":
    main()