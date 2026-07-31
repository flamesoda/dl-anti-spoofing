#!/usr/bin/env python3
"""Validate a two-column submission and optionally compute official EER."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.metrics import compute_eer


def read_protocol(path):
    entries = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            _, utterance_id, _, _, label = line.split()
            entries.append((utterance_id, label))
    return entries


def read_submission(path):
    scores = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if len(row) != 2:
                raise ValueError(f"Line {line_number} must contain exactly 2 columns")
            utterance_id, raw_score = row
            if utterance_id in scores:
                raise ValueError(f"Duplicate utterance id: {utterance_id}")
            score = float(raw_score)
            if not np.isfinite(score):
                raise ValueError(f"Non-finite score for {utterance_id}")
            scores[utterance_id] = score
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()

    protocol = read_protocol(args.protocol)
    scores = read_submission(args.submission)
    expected_ids = {utterance_id for utterance_id, _ in protocol}
    missing = expected_ids - scores.keys()
    extra = scores.keys() - expected_ids
    if missing or extra:
        raise ValueError(
            f"Submission mismatch: {len(missing)} missing, {len(extra)} extra"
        )

    bona = np.array([scores[key] for key, label in protocol if label == "bonafide"])
    spoof = np.array([scores[key] for key, label in protocol if label == "spoof"])
    eer, threshold = compute_eer(bona, spoof)
    print(f"Valid submission: {len(protocol)} unique trials")
    print(f"Official pooled EER: {eer * 100:.6f}%")
    print(f"EER threshold: {threshold:.9f}")


if __name__ == "__main__":
    main()
