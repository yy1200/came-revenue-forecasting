#!/usr/bin/env python3
"""Validate the public release contract against the executable replay wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replay_reference import (  # noqa: E402
    FINAL_PRED_COL,
    REFERENCE,
    TICKERS,
    _runner_command,
)


CONTRACT = ROOT / "release_artifacts/came_release_contract.json"
DYNAMIC_FLAGS = {
    "experiment_config",
    "method_family_label",
    "native_backbone_csv",
    "native_card_table_jsonl",
    "output_dir",
    "regime_aware_base_anchor_proposal_csv",
    "tickers",
}


def _parse_flags(command: list[str]) -> tuple[str, dict[str, str | list[str]]]:
    if len(command) < 4 or command[1] != "-m":
        raise ValueError("runner command must use python -m")
    module = command[2]
    flags: dict[str, str | list[str]] = {}
    index = 3
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected runner token: {token}")
        name = token[2:]
        index += 1
        values: list[str] = []
        while index < len(command) and not command[index].startswith("--"):
            values.append(command[index])
            index += 1
        if not values:
            raise ValueError(f"runner flag has no value: {name}")
        flags[name] = values if name == "tickers" else values[0]
    return module, flags


def validate() -> list[str]:
    findings: list[str] = []
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    command = _runner_command(
        experiment=Path("<EXPERIMENT>"),
        proposal=Path("<PROPOSAL>"),
        output_dir=Path("<OUTPUT>"),
        method_alias="<METHOD_ALIAS>",
        tickers=TICKERS,
    )
    module, flags = _parse_flags(command)
    compatibility_flags = {key: value for key, value in flags.items() if key not in DYNAMIC_FLAGS}

    expected_scalars = {
        "schema_version": "came_release_contract_v2",
        "paper_method": "CAME: Company-Aware Evidence-Memory Experts",
        "compatibility_glossary": "GLOSSARY.md",
        "compatibility_implementation_alias": "came_ram_v1p1_partitioned_residual_retained_20260511",
        "supported_entrypoint": "scripts/run_reference_replay.sh",
        "compatibility_runner_module": module,
        "reference_prediction_artifact": REFERENCE.relative_to(ROOT).as_posix(),
    }
    for key, expected in expected_scalars.items():
        if contract.get(key) != expected:
            findings.append(f"release contract {key} mismatch")
    if contract.get("panel") != TICKERS:
        findings.append("release contract panel/order mismatch")
    if contract.get("compatibility_runner_flags") != compatibility_flags:
        findings.append("release contract runner flags differ from executable wrapper")
    if (contract.get("compatibility_prediction_columns") or {}).get("runner_final") != FINAL_PRED_COL:
        findings.append("release contract runner prediction column mismatch")

    selection = contract.get("selection_protocol") or {}
    if selection.get("development_companies") != ["AAPL", "NVDA", "AVGO"]:
        findings.append("release contract development-company split mismatch")
    if selection.get("selection_through") != "FY2023_Q4":
        findings.append("release contract selection cutoff mismatch")
    if (selection.get("temporal_held_out_start"), selection.get("temporal_held_out_end")) != (
        "FY2024_Q1",
        "FY2025_Q4",
    ):
        findings.append("release contract temporal held-out window mismatch")
    if selection.get("settings_fixed_before_held_out_evaluation") is not True:
        findings.append("release contract must record pre-evaluation setting freeze")

    replay = contract.get("reference_replay") or {}
    expected_replay = {
        "rows": 336,
        "companies": 12,
        "quarters_per_company": 28,
        "temporal_held_out_rows": 96,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-12,
        "final_prediction_inputs": 0,
        "aapl_fy2019_q4_two_stage_input_correction": True,
    }
    if replay != expected_replay:
        findings.append("release contract replay boundary mismatch")
    return findings


def main() -> None:
    findings = validate()
    if findings:
        print("release contract validation failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        raise SystemExit(1)
    print("release contract validation passed (runner flags, panel, split, and replay boundary)")


if __name__ == "__main__":
    main()
