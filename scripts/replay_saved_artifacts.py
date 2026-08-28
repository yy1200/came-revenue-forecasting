#!/usr/bin/env python3
"""Validate normalized traces and recompute frozen headline metrics."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "release_artifacts/normalized_forecast_traces.csv"
EXPECTED = {
    "development_inclusive": {
        "n": 336,
        "companies": 12,
        "came_macro_smape": 0.059252682492693755,
        "anchor_macro_smape": 0.0699501518466293,
        "came_pooled_mae": 1810315072.154405,
        "anchor_pooled_mae": 2250751676.6046786,
    },
    "temporal_held_out": {
        "n": 96,
        "companies": 12,
        "came_macro_smape": 0.037810422421270695,
        "anchor_macro_smape": 0.04717070040280311,
        "came_pooled_mae": 1483160868.8273544,
        "anchor_pooled_mae": 1929407262.189153,
    },
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(rows: list[dict[str, str]], pred_col: str) -> tuple[float, float]:
    company_smapes: dict[str, list[float]] = defaultdict(list)
    absolute_errors = []
    for row in rows:
        actual = float(row["actual"])
        pred = float(row[pred_col])
        absolute_errors.append(abs(pred - actual))
        company_smapes[row["ticker"]].append(2.0 * abs(pred - actual) / max(abs(pred) + abs(actual), 1e-12))
    macro_smape = sum(sum(values) / len(values) for values in company_smapes.values()) / len(company_smapes)
    pooled_mae = sum(absolute_errors) / len(absolute_errors)
    return macro_smape, pooled_mae


def _assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise SystemExit(f"{label}: expected {expected:.15g}, found {actual:.15g}")


def main() -> None:
    rows = _read(TRACES)
    keys = [(row["ticker"], row["quarter"]) for row in rows]
    if len(rows) != 336 or len(set(keys)) != 336:
        raise SystemExit(f"normalized trace coverage mismatch: rows={len(rows)} unique_keys={len(set(keys))}")
    for row in rows:
        for field in ("actual", "final_came_prediction", "online_anchor_prediction"):
            if not math.isfinite(float(row[field])):
                raise SystemExit(f"{row['ticker']} {row['quarter']}: non-finite {field}")

    surfaces = {
        "development_inclusive": rows,
        "temporal_held_out": [
            row for row in rows if "FY2024_Q1" <= row["quarter"] <= "FY2025_Q4"
        ],
    }
    for name, surface in surfaces.items():
        expected = EXPECTED[name]
        companies = {row["ticker"] for row in surface}
        if len(surface) != expected["n"] or len(companies) != expected["companies"]:
            raise SystemExit(f"{name}: coverage mismatch ({len(surface)} rows, {len(companies)} companies)")
        came_smape, came_mae = _metrics(surface, "final_came_prediction")
        anchor_smape, anchor_mae = _metrics(surface, "online_anchor_prediction")
        _assert_close(f"{name} CAME macro sMAPE", came_smape, expected["came_macro_smape"])
        _assert_close(f"{name} anchor macro sMAPE", anchor_smape, expected["anchor_macro_smape"])
        _assert_close(f"{name} CAME pooled MAE", came_mae, expected["came_pooled_mae"])
        _assert_close(f"{name} anchor pooled MAE", anchor_mae, expected["anchor_pooled_mae"])
        print(
            f"{name}: n={len(surface)} companies={len(companies)} "
            f"CAME macro sMAPE={came_smape:.6f} anchor={anchor_smape:.6f} "
            f"CAME pooled MAE={came_mae:.3f} anchor={anchor_mae:.3f}"
        )
    print("normalized trace validation and metric recomputation passed")


if __name__ == "__main__":
    main()
