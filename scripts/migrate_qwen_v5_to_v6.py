#!/usr/bin/env python
"""Repair Qwen v5 confidence weights without rerunning the 72B model.

The v5 extractor already persisted the confidence-derived weight in every
target column.  v6 changes only the calibration of the old low-confidence
state: v5 used 0.85, which was larger than the medium-confidence weight 0.65.
This migration maps v5 low-confidence rows to v6 (probability 0.65/0.35,
weight 0.25), preserves medium/high/not-stated rows, updates row provenance,
and writes a new complete manifest. It is safe to run atomically on a copy or
in-place after the original file has been backed up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

# Allow direct execution as ``python scripts/migrate_...py`` from any cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core import config

OLD_VERSION = "clinical-report-labels-v5"
NEW_VERSION = "clinical-report-labels-v6"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate(input_csv: str, output_csv: str, backup: str | None = None) -> dict:
    manifest_path = f"{input_csv}.manifest.json"
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("extractor_version") != OLD_VERSION:
        raise ValueError(
            f"Expected {OLD_VERSION} manifest, got {manifest.get('extractor_version')!r}"
        )
    if manifest.get("status") != "complete":
        raise ValueError("Refusing to migrate an incomplete extraction")

    frame = pd.read_csv(input_csv)
    if len(frame) != int(manifest.get("completed_studies", -1)):
        raise ValueError("CSV row count does not match the v5 manifest")

    migrated = 0
    for target in config.TARGETS:
        value_col = target
        weight_col = f"{target}_weight"
        if value_col not in frame or weight_col not in frame:
            raise ValueError(f"Missing target columns for {target!r}")
        values = pd.to_numeric(frame[value_col], errors="raise").to_numpy(float, copy=True)
        weights = pd.to_numeric(frame[weight_col], errors="raise").to_numpy(float, copy=True)
        low = np.isclose(weights, 0.85, atol=1e-6) & np.isfinite(values)
        if low.any():
            values[low] = np.where(values[low] >= 0.5, 0.65, 0.35)
            weights[low] = 0.25
            migrated += int(low.sum())
        frame[value_col] = values
        frame[weight_col] = weights

    if "extractor_version" in frame:
        frame["extractor_version"] = NEW_VERSION
    else:
        raise ValueError("Pseudo-label CSV lacks per-row extractor provenance")

    if backup and os.path.abspath(backup) != os.path.abspath(input_csv):
        shutil.copy2(input_csv, backup)

    output_manifest = dict(manifest)
    output_manifest["extractor_version"] = NEW_VERSION
    output_manifest["derived_from_extractor_version"] = OLD_VERSION
    output_manifest["derived_from_pseudo_csv_sha256"] = sha256_file(input_csv)
    output_manifest["migration"] = "qwen-v5-confidence-order-repair"

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    temp_csv = f"{output_csv}.tmp"
    frame.to_csv(temp_csv, index=False)
    os.replace(temp_csv, output_csv)
    output_manifest["pseudo_csv_sha256"] = sha256_file(output_csv)
    manifest_out = f"{output_csv}.manifest.json"
    temp_manifest = f"{manifest_out}.tmp"
    with open(temp_manifest, "w", encoding="utf-8") as stream:
        json.dump(output_manifest, stream, indent=2, sort_keys=True)
    os.replace(temp_manifest, manifest_out)
    return {"rows": len(frame), "low_confidence_rows_repaired": migrated, "output": output_csv}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backup", default=None)
    args = parser.parse_args()
    print(json.dumps(migrate(args.input, args.output, args.backup), indent=2))


if __name__ == "__main__":
    main()
