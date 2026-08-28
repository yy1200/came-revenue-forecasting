#!/usr/bin/env python3
"""Regenerate and verify the 336-row CAME reference prediction surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (ROOT / "output").resolve()
OUTPUT_MARKER_NAME = ".came_reference_replay_output"
OUTPUT_MARKER_CONTENT = "came_reference_replay_v1\n"
INPUT_ROOT = ROOT / "replay_inputs/retained_336"
REFERENCE = ROOT / "release_artifacts/normalized_forecast_traces.csv"
PRE_CORRECTION_ALIAS = "came_reference_pre_correction"
CORRECTED_AAPL_ALIAS = "came_reference_aapl_correction"
FINAL_PRED_COL = "pred_csais_rawcard_direct_experts_v1"
RAM_PROPOSAL_COL = "pred_came_ge_v1p1_noguidance_anchor_guard_candidate"
AAPL_CORRECTION_KEY = ("AAPL", "FY2019_Q4")
AAPL_PRE_CORRECTION_GUID_MID = "32030500000.0"
TICKERS = ["AAPL", "NVDA", "AVGO", "AMZN", "ASML", "GOOGL", "INTC", "META", "MSFT", "MU", "TSLA", "ORCL"]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _prepare_output_dir(requested: Path) -> Path:
    candidate = requested if requested.is_absolute() else ROOT / requested
    output_dir = Path(os.path.abspath(candidate))
    try:
        relative = output_dir.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(f"output directory must be under {OUTPUT_ROOT}: {output_dir}") from exc
    if not relative.parts:
        raise ValueError(f"output directory must be a strict descendant of {OUTPUT_ROOT}")

    current = OUTPUT_ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"output path must not contain symlinks: {current}")

    marker = output_dir / OUTPUT_MARKER_NAME
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path exists and is not a directory: {output_dir}")
        try:
            marker_content = marker.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"refusing to remove unowned output directory: {output_dir}") from exc
        if marker_content != OUTPUT_MARKER_CONTENT:
            raise ValueError(f"refusing to remove unowned output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    marker.write_text(OUTPUT_MARKER_CONTENT, encoding="utf-8")
    return output_dir


def _prepare_pre_correction_inputs(work_dir: Path) -> tuple[Path, Path]:
    base_config = json.loads((INPUT_ROOT / "experiment.json").read_text(encoding="utf-8"))
    config = json.loads(json.dumps(base_config))
    aapl = next(company for company in config["companies"] if company["ticker"] == "AAPL")
    corrected_stat = ROOT / aapl["stat_baseline_predictions_csv"]
    stat_rows, stat_fields = _read_csv(corrected_stat)
    for row in stat_rows:
        if row["fiscal_quarter"] == AAPL_CORRECTION_KEY[1]:
            row["pred__guid_mid"] = AAPL_PRE_CORRECTION_GUID_MID
    pre_correction_stat = work_dir / "AAPL_stat_anchor_candidates_pre_correction.csv"
    _write_csv(pre_correction_stat, stat_rows, stat_fields)
    aapl["stat_baseline_predictions_csv"] = str(pre_correction_stat)
    pre_correction_config = work_dir / "experiment_pre_correction.json"
    _write_json(pre_correction_config, config)

    proposal_rows, proposal_fields = _read_csv(INPUT_ROOT / "ram_proposal_input.csv")
    for row in proposal_rows:
        if (row["ticker"], row["quarter"]) == AAPL_CORRECTION_KEY:
            row[RAM_PROPOSAL_COL] = AAPL_PRE_CORRECTION_GUID_MID
    pre_correction_proposal = work_dir / "ram_proposal_pre_correction.csv"
    _write_csv(pre_correction_proposal, proposal_rows, proposal_fields)
    return pre_correction_config, pre_correction_proposal


def _runner_command(
    *,
    experiment: Path,
    proposal: Path,
    output_dir: Path,
    method_alias: str,
    tickers: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "company_stat_anchor_shock.run_csais_rawcard_direct_experts_v1",
        "--experiment_config",
        str(experiment),
        "--native_backbone_csv",
        str(INPUT_ROOT / "native_backbone.csv"),
        "--native_card_table_jsonl",
        str(INPUT_ROOT / "normalized_native_forward_cards.jsonl"),
        "--tickers",
        *tickers,
        "--method_family_label",
        method_alias,
        "--expert_contract",
        "full",
        "--gate_mode",
        "off",
        "--intrinsic_target_mode",
        "partitioned_residual_v1",
        "--intrinsic_reliability_mode",
        "history_bucket_v1",
        "--intrinsic_explicit_guidance_guard_mode",
        "support_shrink_v1",
        "--intrinsic_temporal_dedup_mode",
        "off",
        "--temporal_score_mode",
        "current",
        "--temporal_support_mode",
        "current",
        "--temporal_direction_mode",
        "off",
        "--temporal_context_guard_mode",
        "derived_weak_min_align",
        "--temporal_context_memory_mode",
        "typed_retrieval",
        "--temporal_evidence_filter_mode",
        "off",
        "--temporal_evidence_filter_scope",
        "all",
        "--temporal_context_quality_mode",
        "off",
        "--temporal_context_quality_weight",
        "0.0",
        "--temporal_context_retrieval_weight",
        "0.35",
        "--temporal_segment_compat_weight",
        "0.12",
        "--temporal_context_support_weight",
        "0.2",
        "--temporal_novelty_shrink_weight",
        "0.35",
        "--temporal_guidance_trust_weight",
        "0.3",
        "--temporal_segment_support_weight",
        "0.2",
        "--temporal_soft_agreement_weight",
        "0.2",
        "--temporal_context_memory_sparsity_power",
        "1.0",
        "--guidance_quality_guardrail_mode",
        "unified_v1",
        "--guidance_expert_mode",
        "explicit_history_trust_non_guidance_anchor_v1",
        "--guidance_expert_min_history",
        "1",
        "--guidance_expert_history_tau",
        "0.2",
        "--guidance_expert_support_scale",
        "1.25",
        "--regime_aware_base_anchor_mode",
        "signed_strength_v1",
        "--regime_aware_base_anchor_proposal_csv",
        str(proposal),
        "--regime_aware_base_anchor_proposal_pred_col",
        RAM_PROPOSAL_COL,
        "--regime_aware_base_anchor_proposal_filter_col",
        "surface",
        "--regime_aware_base_anchor_proposal_filter_value",
        "main_all12_topall",
        "--regime_aware_base_anchor_min_history",
        "6",
        "--regime_aware_base_anchor_signed_strength_threshold",
        "0.02",
        "--regime_aware_base_anchor_upward_ratio",
        "1.0",
        "--anchor_selection_mode",
        "online_historical_mae",
        "--anchor_online_min_history",
        "4",
        "--anchor_online_score_metric",
        "smape",
        "--anchor_online_window",
        "24",
        "--anchor_online_half_life",
        "0.0",
        "--anchor_online_same_quarter_weight",
        "0.0",
        "--anchor_guidance_regime_mode",
        "off",
        "--anchor_explicit_guidance_proximity_mode",
        "off",
        "--anchor_robust_momentum_mode",
        "off",
        "--coverage_drift_mode",
        "fail",
        "--missing_anchor_policy",
        "skip",
        "--arbitration_mode",
        "current",
        "--output_dir",
        str(output_dir),
    ]


def _run(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CAME_REFERENCE_REPLAY"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _is_close(actual: float, expected: float, abs_tol: float, rel_tol: float) -> bool:
    return math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/reference_replay")
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--rel-tol", type=float, default=1e-12)
    args = parser.parse_args()
    try:
        output_dir = _prepare_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    validator = ROOT / "scripts/validate_reference_inputs.py"
    contract_validator = ROOT / "scripts/validate_release_contract.py"
    subprocess.run([sys.executable, str(contract_validator)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(validator)], cwd=ROOT, check=True)
    pre_correction_config, pre_correction_proposal = _prepare_pre_correction_inputs(output_dir / "prepared_inputs")
    pre_correction_dir = output_dir / "pre_correction_runner"
    pre_correction_command = _runner_command(
        experiment=pre_correction_config,
        proposal=pre_correction_proposal,
        output_dir=pre_correction_dir,
        method_alias=PRE_CORRECTION_ALIAS,
        tickers=TICKERS,
    )
    _run(pre_correction_command, output_dir / "pre_correction_runner.log")

    corrected_aapl_dir = output_dir / "corrected_aapl_runner"
    corrected_aapl_command = _runner_command(
        experiment=INPUT_ROOT / "experiment.json",
        proposal=INPUT_ROOT / "ram_proposal_input.csv",
        output_dir=corrected_aapl_dir,
        method_alias=CORRECTED_AAPL_ALIAS,
        tickers=["AAPL"],
    )
    _run(corrected_aapl_command, output_dir / "corrected_aapl_runner.log")

    pre_correction_rows, _ = _read_csv(pre_correction_dir / f"{PRE_CORRECTION_ALIAS}_quarterly.csv")
    corrected_aapl_rows, _ = _read_csv(corrected_aapl_dir / f"{CORRECTED_AAPL_ALIAS}_quarterly.csv")
    reference_rows, _ = _read_csv(REFERENCE)
    pre_correction = {(row["ticker"], row["quarter"]): row for row in pre_correction_rows}
    corrected_aapl = {(row["ticker"], row["quarter"]): row for row in corrected_aapl_rows}
    reference = {(row["ticker"], row["quarter"]): row for row in reference_rows}
    if len(pre_correction) != 336 or len(corrected_aapl) != 28 or len(reference) != 336:
        raise SystemExit(
            f"coverage failure: pre_correction={len(pre_correction)} corrected_aapl={len(corrected_aapl)} reference={len(reference)}"
        )

    comparison_rows: list[dict[str, Any]] = []
    mismatches = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for key in sorted(reference):
        stage = "corrected_aapl_runner" if key == AAPL_CORRECTION_KEY else "pre_correction_runner"
        generated_row = corrected_aapl[key] if key == AAPL_CORRECTION_KEY else pre_correction[key]
        generated = float(generated_row[FINAL_PRED_COL])
        expected = float(reference[key]["final_came_prediction"])
        abs_diff = abs(generated - expected)
        rel_diff = abs_diff / max(abs(expected), 1.0)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        matched = _is_close(generated, expected, args.abs_tol, args.rel_tol)
        if not matched:
            mismatches.append(
                {"ticker": key[0], "quarter": key[1], "generated": generated, "expected": expected, "abs_diff": abs_diff}
            )
        comparison_rows.append(
            {
                "ticker": key[0],
                "quarter": key[1],
                "generation_stage": stage,
                "reconstructed_prediction": format(generated, ".17g"),
                "reference_prediction": format(expected, ".17g"),
                "absolute_difference": format(abs_diff, ".17g"),
                "within_tolerance": int(matched),
            }
        )

    comparison_path = output_dir / "reference_replay_comparison.csv"
    _write_csv(
        comparison_path,
        comparison_rows,
        [
            "ticker",
            "quarter",
            "generation_stage",
            "reconstructed_prediction",
            "reference_prediction",
            "absolute_difference",
            "within_tolerance",
        ],
    )
    result = {
        "status": "pass" if not mismatches else "fail",
        "runner_module": "company_stat_anchor_shock.run_csais_rawcard_direct_experts_v1",
        "compatibility_prediction_column": FINAL_PRED_COL,
        "reference_prediction_column": "final_came_prediction",
        "reference_prediction_artifact": "release_artifacts/normalized_forecast_traces.csv",
        "rows": 336,
        "companies": 12,
        "pre_correction_runner_rows_used": 335,
        "corrected_aapl_runner_rows_used": 1,
        "anchor_memory_proposal_role": "frozen upstream no-guidance proposal consumed by the compatibility runner; not a final CAME prediction",
        "final_prediction_inputs": 0,
        "abs_tolerance": args.abs_tol,
        "rel_tolerance": args.rel_tol,
        "matched_predictions": 336 - len(mismatches),
        "mismatched_predictions": len(mismatches),
        "max_absolute_difference": max_abs_diff,
        "max_relative_difference": max_rel_diff,
        "mismatch_examples": mismatches[:20],
        "outputs": {
            "comparison_csv": str(comparison_path.relative_to(ROOT)),
            "pre_correction_runner_log": str((output_dir / "pre_correction_runner.log").relative_to(ROOT)),
            "corrected_aapl_runner_log": str((output_dir / "corrected_aapl_runner.log").relative_to(ROOT)),
        },
    }
    _write_json(output_dir / "reference_replay_validation.json", result)
    print(
        f"CAME reference replay: status={result['status']} rows=336 companies=12 "
        f"matched={result['matched_predictions']} mismatched={result['mismatched_predictions']} "
        f"max_abs_diff={max_abs_diff:.12g} max_rel_diff={max_rel_diff:.12g} "
        "runner_generated=336 final_prediction_inputs=0"
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
