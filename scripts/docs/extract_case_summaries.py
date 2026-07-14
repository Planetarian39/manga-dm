"""Export a small, reviewable summary from allowlisted per-galaxy posteriors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _summary(values: np.ndarray) -> dict[str, float]:
    q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "q16": round(float(q16), 4),
        "median": round(float(median), 4),
        "q84": round(float(q84), 4),
    }


def summarize_file(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as dataset:
        galaxy_id = str(dataset.attrs.get("plate_ifu", path.name.split("_nfw_")[0]))
        log10_m200 = np.asarray(dataset["log10_M200_samples"], dtype=float)
        log10_c = np.asarray(dataset["log10_c_samples"], dtype=float)
        if log10_m200.shape != log10_c.shape or log10_m200.size == 0:
            raise ValueError(f"Posterior arrays must be non-empty and aligned: {path}")
        sample_count = int(np.asarray(dataset.get("sample_count", log10_m200.size)))

    return {
        "galaxyId": galaxy_id,
        "sampleCount": sample_count,
        "log10M200": _summary(log10_m200),
        "log10C": _summary(log10_c),
        "correlation": round(float(np.corrcoef(log10_m200, log10_c)[0, 1]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [summarize_file(path) for path in sorted(args.input_dir.glob("*.nc"))]
    payload = {
        "schemaVersion": 1,
        "description": (
            "Per-galaxy descriptive posterior summaries for method demonstration only; "
            "these values are not an aggregate scientific result."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} case summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
