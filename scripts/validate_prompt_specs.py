#!/usr/bin/env python3
"""Validate prompt-spec hashes, exact render contracts, and release boundaries."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompt_specs.executable.came_evidence_card_extraction import (
    EVENT_FALLBACK_DOCUMENT_CHAR_LIMIT,
    EVENT_FALLBACK_SYSTEM_SUFFIX,
    EVENT_TEMPERATURE,
    PRIMARY_DOCUMENT_CHAR_LIMIT,
    RETAINED_EVENT_POLICY,
    build_event_fallback_system_prompt,
    build_event_request_spec,
    build_event_system_prompt,
    build_event_user_prompt,
    build_relation_request_spec,
    build_relation_system_prompt,
    build_relation_user_prompt,
    select_event_fallback_document,
)
from prompt_specs.executable.direct_same_task_forecast import (
    build_system_msg,
    build_user_prompt_hist,
    build_user_prompt_hist_guid,
    json_schema_response_format,
    parse_response,
    unit_description,
)


SPEC_ROOT = ROOT / "prompt_specs"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def validate() -> list[str]:
    findings: list[str] = []
    manifest_path = SPEC_ROOT / "spec_manifest.json"
    if not manifest_path.is_file():
        return ["missing prompt_specs/spec_manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_files = manifest.get("files") or {}
    actual_files = {
        path.relative_to(SPEC_ROOT).as_posix()
        for path in SPEC_ROOT.rglob("*")
        if path.is_file() and path.name != "spec_manifest.json" and "__pycache__" not in path.parts
    }
    if actual_files != set(expected_files):
        findings.append(
            f"prompt-spec file set mismatch: only_actual={sorted(actual_files - set(expected_files))} "
            f"only_manifest={sorted(set(expected_files) - actual_files)}"
        )
    for relative, expected_hash in expected_files.items():
        path = SPEC_ROOT / relative
        if path.is_file() and _sha256_bytes(path.read_bytes()) != expected_hash:
            findings.append(f"prompt_specs/{relative}: SHA-256 mismatch")

    source_refs = []
    for section in ("direct_same_task_forecast", "came_evidence_card_extraction"):
        source_refs.extend((manifest.get(section, {}).get("source_artifact_hashes") or {}).items())
    for key, value in source_refs:
        if key.endswith("_sha256") and not HEX64.fullmatch(str(value)):
            findings.append(f"spec_manifest.json: invalid source hash {key}")

    direct_schema = json.loads(
        (SPEC_ROOT / "schemas/direct_revenue_forecast_response.schema.json").read_text(encoding="utf-8")
    )
    response_schema = json_schema_response_format()["json_schema"]["schema"]
    direct_schema_without_metadata = {
        key: value for key, value in direct_schema.items() if key not in {"$schema", "title"}
    }
    if response_schema != direct_schema_without_metadata:
        findings.append("direct response schema differs from executable response_format")

    unit_desc, scale = unit_description("billion", "EUR")
    if (unit_desc, scale) != ("billion EUR", 1e9):
        findings.append("reporting-currency unit contract failed")
    history = "fiscal_quarter,revenue\nFY2023_Q4,10.0\n"
    guidance = "fiscal_quarter,guid_low,guid_high,guid_mid,pct\nFY2024_Q1,10,12,11,9.0909\n"
    system = build_system_msg("Example Company")
    hist_prompt = build_user_prompt_hist(history, "FY2024_Q1", unit_desc, "Example Company")
    guid_prompt = build_user_prompt_hist_guid(history, guidance, "FY2024_Q1", unit_desc, "Example Company")
    if "Example Company" not in system or "billion EUR" not in hist_prompt:
        findings.append("direct company/currency placeholder contract failed")
    if "revenue is unknown" not in guid_prompt or "FY2024_Q1" not in guid_prompt:
        findings.append("History+Guidance target-hiding contract failed")
    if parse_response('{"fiscal_quarter":"FY2024_Q1","pred_revenue":11.5}', "FY2024_Q1")["pred_revenue"] != 11.5:
        findings.append("direct parser valid-response contract failed")
    for invalid in (
        '{"fiscal_quarter":"FY2024_Q2","pred_revenue":11.5}',
        '{"fiscal_quarter":"FY2024_Q1","pred_revenue":0}',
    ):
        try:
            parse_response(invalid, "FY2024_Q1")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            findings.append("direct parser accepted an invalid target or prediction")

    relation = build_relation_system_prompt(["Example::Segment"], ["BOOSTS_REVENUE_OF"])
    event = build_event_system_prompt(["Example::Segment"], "FY2023_Q4", "FY2024_Q1")
    if RETAINED_EVENT_POLICY != "strict_target_quarter_causal_driver" or "DRIVER vs OUTCOME" not in event:
        findings.append("strict target-quarter causal-driver event contract missing")
    if "separately stated causal driver is NOT required" in event:
        findings.append("diagnostic relaxed event policy leaked into retained prompt export")
    if "Example::Segment" not in relation or "BOOSTS_REVENUE_OF" not in relation:
        findings.append("CAME relation placeholders were not rendered")
    if "FY2023_Q4" not in event or "FY2024_Q1" not in event:
        findings.append("CAME event quarter placeholders were not rendered")

    relation_user = build_relation_user_prompt("<DOCUMENT>", "<OBSERVED_QUARTER>")
    event_user = build_event_user_prompt("<DOCUMENT>", "<OBSERVED_QUARTER>")
    if relation_user != "Transcript (<OBSERVED_QUARTER>):\n<DOCUMENT>":
        findings.append("CAME relation user-message wrapper mismatch")
    if event_user != "Transcript (<OBSERVED_QUARTER>):\n\n<DOCUMENT>":
        findings.append("CAME event user-message wrapper mismatch")
    if build_relation_user_prompt("A\udcffB", "FY2024_Q1") != "Transcript (FY2024_Q1):\nAB":
        findings.append("CAME JSON-safe document-text contract failed")
    relation_request = build_relation_request_spec(
        "<DOCUMENT>", "<OBSERVED_QUARTER>", ["<ALLOWED_SEGMENT>"], ["<ALLOWED_RELATION>"]
    )
    event_request = build_event_request_spec(
        "<DOCUMENT>", "<OBSERVED_QUARTER>", "<TARGET_QUARTER>", ["<ALLOWED_SEGMENT>"]
    )
    if "temperature" in relation_request:
        findings.append("CAME relation request must omit the temperature parameter")
    if event_request.get("temperature") != EVENT_TEMPERATURE:
        findings.append("CAME event request temperature mismatch")
    if relation_request.get("response_format_model") != "ClaimOutput":
        findings.append("CAME relation response-model contract mismatch")
    if event_request.get("response_format_model") != "EventClaimOutput":
        findings.append("CAME event response-model contract mismatch")
    if len(build_relation_user_prompt("x" * (PRIMARY_DOCUMENT_CHAR_LIMIT + 1), "FY2024_Q1").rsplit("\n", 1)[-1]) != PRIMARY_DOCUMENT_CHAR_LIMIT:
        findings.append("CAME primary document truncation contract failed")

    synthetic_document = (
        "Context sentence. We expect revenue in Q1 to grow due to demand. "
        "Closing sentence. What about analyst demand?"
    )
    selected_fallback = select_event_fallback_document(synthetic_document, "FY2024_Q1")
    if selected_fallback != "Context sentence.\nWe expect revenue in Q1 to grow due to demand.\nClosing sentence.":
        findings.append("CAME strict-driver fallback selection contract failed")
    fallback_request = build_event_request_spec(
        synthetic_document,
        "FY2023_Q4",
        "FY2024_Q1",
        ["Example::Segment"],
        fallback=True,
    )
    fallback_system = fallback_request["messages"][0]["content"]
    if not str(fallback_system).endswith(EVENT_FALLBACK_SYSTEM_SUFFIX):
        findings.append("CAME strict-driver fallback suffix contract failed")
    fallback_user = str(fallback_request["messages"][1]["content"])
    if len(fallback_user.rsplit("\n\n", 1)[-1]) > EVENT_FALLBACK_DOCUMENT_CHAR_LIMIT:
        findings.append("CAME fallback document truncation contract failed")

    canonical = {
        "direct_system_company_placeholder": build_system_msg("<COMPANY>"),
        "direct_history_placeholders": build_user_prompt_hist(
            "<HISTORY_CSV>", "<TARGET_QUARTER>", "<UNIT_AND_REPORTING_CURRENCY>", "<COMPANY>"
        ),
        "direct_history_guidance_placeholders": build_user_prompt_hist_guid(
            "<HISTORY_WITH_GUIDANCE_CSV>",
            "<TARGET_GUIDANCE_CSV>",
            "<TARGET_QUARTER>",
            "<UNIT_AND_REPORTING_CURRENCY>",
            "<COMPANY>",
        ),
        "came_relation_placeholders": build_relation_system_prompt(
            ["<ALLOWED_SEGMENT>"], ["<ALLOWED_RELATION>"]
        ),
        "came_event_strict_driver_placeholders": build_event_system_prompt(
            ["<ALLOWED_SEGMENT>"], "<OBSERVED_QUARTER>", "<TARGET_QUARTER>"
        ),
        "came_relation_user_placeholders": relation_user,
        "came_event_user_placeholders": event_user,
        "came_event_strict_driver_fallback_placeholders": build_event_fallback_system_prompt(
            ["<ALLOWED_SEGMENT>"], "<OBSERVED_QUARTER>", "<TARGET_QUARTER>"
        ),
    }
    recorded_hashes = manifest.get("canonical_render_sha256") or {}
    for key, rendered in canonical.items():
        if _sha256_text(rendered) != recorded_hashes.get(key):
            findings.append(f"canonical prompt render hash mismatch: {key}")

    forbidden_names = re.compile(r"(?i)(response|cache|secret|api[_-]?output)")
    for relative in actual_files:
        if forbidden_names.search(Path(relative).name) and relative != "schemas/direct_revenue_forecast_response.schema.json":
            findings.append(f"forbidden prompt-package member name: {relative}")
    return findings


def main() -> None:
    findings = validate()
    if findings:
        print("prompt-spec validation failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        raise SystemExit(1)
    manifest = json.loads((SPEC_ROOT / "spec_manifest.json").read_text(encoding="utf-8"))
    print(
        "prompt-spec validation passed "
        f"(files={len(manifest['files'])} exact_contracts={len(manifest['contract_classes']['exact_executable'])})"
    )


if __name__ == "__main__":
    main()
