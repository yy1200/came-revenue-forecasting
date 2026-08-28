#!/usr/bin/env python3
"""Validate integrity and the no-final-prediction boundary of reference inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "replay_inputs/retained_336"
TOKEN_RE = re.compile(r"^[0-9a-f]{24}$")
ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s'\"])(?:/home/|/Users/|[A-Za-z]:\\Users\\)")
DIRECT_TEXT_KEYS = {
    "api_output",
    "evidence",
    "excerpt",
    "filing_text",
    "paragraph",
    "memory_verbatim",
    "query_verbatim",
    "quote",
    "raw_text",
    "source_span",
    "source_text",
    "transcript_text",
    "verbatim",
}
PSEUDONYM_KEYS = {
    "attribution_anchor",
    "matched_attribution_anchor",
    "query_attribution_anchor",
    "instance_id",
    "source_record_id",
    "signature_key",
    "query_source_ref",
    "memory_source_ref",
}
FINAL_PREDICTION_NAMES = {
    "final_came_prediction",
    "pred_csais_rawcard_direct_experts_v1",
    "pred_came_ram_v1_signed_strength_guard_t0p02",
    "reference_prediction",
    "retained_prediction",
}
ALLOWED_PROPOSAL_NAME = "pred_came_ge_v1p1_noguidance_anchor_guard_candidate"
REPLAY_MANIFEST_PROVENANCE = {
    "public_release_status": "included_in_public_release",
    "derived_artifact_status": "author_created_release_manifest",
    "source_material_access_status": "not_required_for_public_post_extraction_replay",
    "source_material_redistribution_status": "not_redistributed_no_rights_grant",
    "verbatim_text_status": "withheld_or_absent_quote_free",
}
REPLAY_INPUT_PROVENANCE = {
    "public_release_status": "included_in_public_release",
    "derived_artifact_status": "author_created_derived_replay_input",
    "source_material_access_status": "not_required_for_public_post_extraction_replay",
    "source_material_redistribution_status": "not_redistributed_no_rights_grant",
    "verbatim_text_status": "withheld_or_absent_quote_free",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_json(value: Any, path: str = "$") -> Iterable[str]:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            lower = key.lower()
            child_path = f"{path}.{key}"
            if lower in DIRECT_TEXT_KEYS and isinstance(child, str) and child.strip():
                yield f"{child_path}: direct source text"
            if lower == "factor" and "canonical_factor" in value and isinstance(child, str) and child.strip():
                yield f"{child_path}: raw card factor must be removed when canonical_factor exists"
            if lower in PSEUDONYM_KEYS and isinstance(child, str) and child.strip() and not TOKEN_RE.fullmatch(child):
                yield f"{child_path}: source-derived identifier or anchor must be opaque 24-character hex"
            if lower in FINAL_PREDICTION_NAMES:
                yield f"{child_path}: final CAME prediction is forbidden as replay input"
            if lower == "release_token_ids":
                if not isinstance(child, list) or any(not TOKEN_RE.fullmatch(str(token)) for token in child):
                    yield f"{child_path}: invalid opaque token IDs"
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]")


def validate(input_root: Path = INPUT_ROOT) -> list[str]:
    findings: list[str] = []
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        return ["missing replay_inputs/retained_336/manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "came_retained_336_replay_inputs_v2":
        findings.append("manifest.json: expected came_retained_336_replay_inputs_v2")
    if manifest.get("artifact_provenance") != REPLAY_MANIFEST_PROVENANCE:
        findings.append("manifest.json: replay-manifest provenance profile mismatch")
    if manifest.get("replay_input_provenance") != REPLAY_INPUT_PROVENANCE:
        findings.append("manifest.json: replay-input provenance profile mismatch")
    if manifest.get("provenance_data_dictionary") != "release_artifacts/provenance_data_dictionary.json":
        findings.append("manifest.json: provenance data-dictionary pointer mismatch")
    if manifest.get("provenance_label_counts") != {
        "manifest_bound_files": 401,
        "native_forward_card_rows": 1227,
        "typed_retrieval_card_rows": 2416,
    }:
        findings.append("manifest.json: provenance label counts mismatch")
    if "source_text_policy" in manifest:
        findings.append("manifest.json: ambiguous source_text_policy must not remain")
    expected_files = manifest.get("files") or {}
    for relative, record in expected_files.items():
        path = input_root / relative
        if not path.is_file():
            findings.append(f"{relative}: missing manifest-bound input")
            continue
        actual_hash = _sha256(path)
        if actual_hash != str(record.get("sha256") or ""):
            findings.append(f"{relative}: SHA-256 mismatch")
    actual_files = {
        path.relative_to(input_root).as_posix()
        for path in input_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(expected_files):
        findings.append(
            f"manifest file set mismatch: only_actual={sorted(actual_files - set(expected_files))} "
            f"only_manifest={sorted(set(expected_files) - actual_files)}"
        )

    native_rows = 0
    retrieval_files = 0
    proposal_rows = 0
    company_dirs = set()
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(input_root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{rel}: non-text replay input")
            continue
        if ABSOLUTE_PATH_RE.search(text):
            findings.append(f"{rel}: local absolute path")
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                native_rows += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    findings.append(f"{rel}:{line_number}: invalid JSON: {exc}")
                    continue
                findings.extend(f"{rel}:{line_number}: {item}" for item in _walk_json(payload))
        elif path.suffix == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                findings.append(f"{rel}: invalid JSON: {exc}")
                continue
            findings.extend(f"{rel}: {item}" for item in _walk_json(payload))
            if rel.parts and rel.parts[0] == "retrieve":
                retrieval_files += 1
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            blocked = sorted(set(fields) & FINAL_PREDICTION_NAMES)
            if blocked:
                findings.append(f"{rel}: forbidden final-prediction columns {blocked}")
            direct = sorted(set(field.lower() for field in fields) & DIRECT_TEXT_KEYS)
            if direct:
                findings.append(f"{rel}: direct-text columns {direct}")
            if path.name == "ram_proposal_input.csv":
                proposal_rows = len(rows)
                pred_fields = [field for field in fields if field.startswith("pred_")]
                if pred_fields != [ALLOWED_PROPOSAL_NAME]:
                    findings.append(f"{rel}: expected only the documented RAM proposal column, found {pred_fields}")
            if len(rel.parts) >= 3 and rel.parts[0] == "companies":
                company_dirs.add(rel.parts[1])

    if native_rows != int(manifest.get("native_forward_card_rows") or -1):
        findings.append(f"native forward-card row mismatch: {native_rows}")
    if retrieval_files != 336:
        findings.append(f"typed retrieval coverage mismatch: {retrieval_files} files")
    if proposal_rows != 336:
        findings.append(f"RAM proposal coverage mismatch: {proposal_rows} rows")
    if len(company_dirs) != 12:
        findings.append(f"company input coverage mismatch: {len(company_dirs)} companies")
    return findings


def main() -> None:
    findings = validate()
    if findings:
        print("reference replay-input validation failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        raise SystemExit(1)
    manifest = json.loads((INPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    print(
        "reference replay-input validation passed "
        f"(files={len(manifest['files'])} native_cards={manifest['native_forward_card_rows']} "
        f"retrieval_files={manifest['typed_retrieval_files']} proposal_rows={manifest['ram_proposal_boundary']['rows']}; "
        "final_prediction_inputs=0)"
    )


if __name__ == "__main__":
    main()
