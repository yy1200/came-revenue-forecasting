#!/usr/bin/env python3
"""Fail when release data contains source text, secrets, or local identifiers."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", "dist", "output", "outputs"}
DATA_ROOTS = (
    ROOT / "release_artifacts",
    ROOT / "replay_inputs",
)
DIRECT_TEXT_KEYS = {
    "api_output",
    "card_evidence",
    "driver_evidence",
    "evidence",
    "evidence_preview",
    "excerpt",
    "filing_excerpt",
    "filing_text",
    "guidance_text",
    "input_card_evidence",
    "matched_evidence",
    "memory_evidence",
    "memory_verbatim",
    "paragraph",
    "query_evidence",
    "query_verbatim",
    "quote",
    "raw_text",
    "source_span",
    "source_text",
    "top_evidence",
    "transcript_paragraph",
    "transcript_text",
    "verbatim",
}
BLOCKED_MEMBER_PARTS = {".env", ".git", "output", "outputs"}
BLOCKED_NAME_RE = re.compile(r"(?i)(raw[_-]?(transcript|filing|api)|api[_-]?output|sec[_-]?dump)")
TOKEN_RE = re.compile(r"^[0-9a-f]{24}$")
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
HTML_JSON_SCRIPT_RE = re.compile(
    r'<script\b[^>]*\btype=["\']application/json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]
ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s'\"])(?:/home/|/Users/|[A-Za-z]:\\Users\\)")
PROVENANCE_FIELDS = (
    "public_release_status",
    "derived_artifact_status",
    "source_material_access_status",
    "source_material_redistribution_status",
    "verbatim_text_status",
)
CARD_PROVENANCE = {
    "public_release_status": "included_in_public_release",
    "derived_artifact_status": "author_created_derived_post_extraction_card",
    "source_material_access_status": "not_included_mixed_public_or_provider_access",
    "source_material_redistribution_status": "not_redistributed_no_rights_grant",
    "verbatim_text_status": "withheld_hash_locator_only",
}
TRACE_PROVENANCE = {
    "public_release_status": "included_in_public_release",
    "derived_artifact_status": "author_created_derived_forecast_trace",
    "source_material_access_status": "not_required_for_public_post_extraction_replay",
    "source_material_redistribution_status": "not_redistributed_no_rights_grant",
    "verbatim_text_status": "absent",
}
REPLAY_MANIFEST_PROVENANCE = {
    "public_release_status": "included_in_public_release",
    "derived_artifact_status": "author_created_release_manifest",
    "source_material_access_status": "not_required_for_public_post_extraction_replay",
    "source_material_redistribution_status": "not_redistributed_no_rights_grant",
    "verbatim_text_status": "withheld_or_absent_quote_free",
}


def _permission_record(container: dict[str, Any]) -> bool:
    return bool(container.get("permission_record_id")) and str(container.get("verbatim_text_status")) == "included_permission_confirmed"


def _walk_json(value: Any, path: str = "$", parent: dict[str, Any] | None = None) -> Iterable[str]:
    if isinstance(value, dict):
        permitted = _permission_record(value)
        for key, child in value.items():
            lower_key = str(key).lower()
            child_path = f"{path}.{key}"
            if lower_key in DIRECT_TEXT_KEYS and isinstance(child, str) and child.strip() and not permitted:
                yield f"{child_path}: non-empty direct-text field"
            if lower_key == "factor" and "canonical_factor" in value and isinstance(child, str) and child.strip():
                yield f"{child_path}: raw card factor must be removed when canonical_factor exists"
            if lower_key in PSEUDONYM_KEYS and isinstance(child, str) and child.strip() and not TOKEN_RE.fullmatch(child):
                yield f"{child_path}: source-derived identifier or anchor must be opaque 24-character hex"
            if lower_key == "release_token_ids":
                if not isinstance(child, list) or any(not TOKEN_RE.fullmatch(str(token)) for token in child):
                    yield f"{child_path}: release token IDs must be opaque 24-character hex values"
            if (
                lower_key in {"content", "detail", "label", "text"}
                and isinstance(child, str)
                and len(child.split()) >= 18
                and any(name in value for name in ("segment", "polarity", "relation_family", "driver_source"))
                and not permitted
            ):
                yield f"{child_path}: source-like prose in a structured evidence object"
            yield from _walk_json(child, child_path, value)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]", parent)


def _load_json_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(path.read_text(encoding="utf-8"))]


def _scan_json_file(path: Path) -> list[str]:
    findings = []
    try:
        records = _load_json_records(path)
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{path.relative_to(ROOT)}: invalid JSON: {exc}"]
    for index, record in enumerate(records):
        prefix = f"$[{index}]" if len(records) > 1 else "$"
        findings.extend(f"{path.relative_to(ROOT)}: {item}" for item in _walk_json(record, prefix))
    return findings


def _scan_csv_file(path: Path) -> list[str]:
    findings = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            for key, raw in row.items():
                if raw is None or not raw.strip():
                    continue
                if str(key).lower() in DIRECT_TEXT_KEYS:
                    findings.append(f"{path.relative_to(ROOT)}:{row_number}:{key}: non-empty direct-text column")
                stripped = raw.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    findings.extend(
                        f"{path.relative_to(ROOT)}:{row_number}:{key}: {item}"
                        for item in _walk_json(payload)
                    )
    return findings


def _scan_markdown_file(path: Path) -> list[str]:
    findings = []
    in_sensitive_table = False
    saw_separator = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip().lower() for cell in stripped.strip("|").split("|")]
            if any(cell in {"evidence", "evidence quote", "query evidence", "memory evidence", "verbatim", "quote"} for cell in cells):
                in_sensitive_table = True
                saw_separator = False
                continue
            if in_sensitive_table and all(set(cell) <= {"-", ":", " "} for cell in cells):
                saw_separator = True
                continue
            if in_sensitive_table and saw_separator and "quote_withheld_third_party" not in stripped:
                findings.append(f"{path.relative_to(ROOT)}:{line_number}: quote-bearing Markdown table row")
        elif in_sensitive_table and not stripped:
            in_sensitive_table = False
            saw_separator = False
    return findings


def _scan_html_file(path: Path) -> list[str]:
    findings = []
    text = path.read_text(encoding="utf-8")
    matches = list(HTML_JSON_SCRIPT_RE.finditer(text))
    for index, match in enumerate(matches):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            findings.append(f"{path.relative_to(ROOT)}: embedded application/json[{index}] is invalid: {exc}")
            continue
        findings.extend(
            f"{path.relative_to(ROOT)}: application/json[{index}]: {item}"
            for item in _walk_json(payload)
        )
    return findings


def _text_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in {".gz", ".png", ".jpg", ".jpeg", ".pdf", ".pyc"}:
            continue
        yield path


def _scan_checkout_hygiene() -> list[str]:
    findings = []
    for path in _text_files():
        rel = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ABSOLUTE_PATH_RE.search(text):
            findings.append(f"{rel}: local absolute path")
        if EMAIL_RE.search(text):
            findings.append(f"{rel}: email-like private identifier; release metadata uses names and ORCIDs only")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{rel}: credential-like value")
                break
    return findings


def _validate_required_surfaces() -> list[str]:
    findings = []
    required = [
        ROOT / "LICENSE",
        ROOT / "DATA_LICENSE.md",
        ROOT / "CITATION.cff",
        ROOT / "GLOSSARY.md",
        ROOT / "release_artifacts/normalized_evidence_cards.csv",
        ROOT / "release_artifacts/normalized_forecast_traces.csv",
        ROOT / "release_artifacts/release_manifest.json",
        ROOT / "release_artifacts/provenance_data_dictionary.json",
        ROOT / "release_artifacts/came_release_contract.json",
        ROOT / "release_artifacts/schemas/normalized_evidence_card.schema.json",
        ROOT / "release_artifacts/schemas/normalized_forecast_trace.schema.json",
        ROOT / "replay_inputs/retained_336/manifest.json",
        ROOT / "replay_inputs/retained_336/normalized_native_forward_cards.jsonl",
        ROOT / "replay_inputs/retained_336/ram_proposal_input.csv",
        ROOT / "prompt_specs/spec_manifest.json",
        ROOT / "prompt_specs/executable/direct_same_task_forecast.py",
        ROOT / "prompt_specs/executable/came_evidence_card_extraction.py",
    ]
    for path in required:
        if not path.is_file():
            findings.append(f"missing required release file: {path.relative_to(ROOT)}")
    trace_path = ROOT / "release_artifacts/normalized_forecast_traces.csv"
    card_path = ROOT / "release_artifacts/normalized_evidence_cards.csv"
    card_ids: set[str] = set()
    if card_path.is_file():
        with card_path.open(encoding="utf-8", newline="") as handle:
            card_rows = list(csv.DictReader(handle))
        card_ids = {row["card_id"] for row in card_rows}
        if len(card_rows) != 1227 or len(card_ids) != 1227:
            findings.append(
                "release_artifacts/normalized_evidence_cards.csv: "
                f"expected 1227 unique cards, found rows={len(card_rows)} unique={len(card_ids)}"
            )
    if trace_path.is_file():
        with trace_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(row["ticker"], row["quarter"]) for row in rows}
        if len(rows) != 336 or len(keys) != 336:
            findings.append(
                "release_artifacts/normalized_forecast_traces.csv: "
                f"expected 336 unique rows, found rows={len(rows)} unique={len(keys)}"
            )
        for row_number, row in enumerate(rows, start=2):
            try:
                linked = json.loads(row["normalized_card_ids_json"])
            except (json.JSONDecodeError, KeyError) as exc:
                findings.append(f"release_artifacts/normalized_forecast_traces.csv:{row_number}: invalid card links: {exc}")
                continue
            if int(row.get("normalized_card_count") or -1) != len(linked):
                findings.append(f"release_artifacts/normalized_forecast_traces.csv:{row_number}: card count mismatch")
            missing = sorted(set(linked) - card_ids)
            if missing:
                findings.append(
                    f"release_artifacts/normalized_forecast_traces.csv:{row_number}: "
                    f"unknown card IDs {missing[:5]}"
                )
    return findings


def _validate_complete_schemas() -> list[str]:
    findings: list[str] = []
    surfaces = (
        (
            ROOT / "release_artifacts/normalized_evidence_cards.csv",
            ROOT / "release_artifacts/schemas/normalized_evidence_card.schema.json",
        ),
        (
            ROOT / "release_artifacts/normalized_forecast_traces.csv",
            ROOT / "release_artifacts/schemas/normalized_forecast_trace.schema.json",
        ),
    )
    for csv_path, schema_path in surfaces:
        if not csv_path.is_file() or not schema_path.is_file():
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        relative = schema_path.relative_to(ROOT)
        if list(properties) != fields or required != fields:
            findings.append(f"{relative}: properties and required fields must exactly match CSV header order")
        if schema.get("additionalProperties") is not False:
            findings.append(f"{relative}: additionalProperties must be false")
        for field in fields:
            definition = properties.get(field) or {}
            if not str(definition.get("description") or "").strip():
                findings.append(f"{relative}: missing description for {field}")
                continue
            allowed = definition.get("enum")
            constant = definition.get("const")
            pattern = definition.get("pattern")
            for row_number, row in enumerate(rows, start=2):
                value = str(row.get(field) or "")
                if allowed is not None and value not in {str(item) for item in allowed}:
                    findings.append(f"{csv_path.relative_to(ROOT)}:{row_number}:{field}: value outside schema enum")
                    break
                if constant is not None and value != str(constant):
                    findings.append(f"{csv_path.relative_to(ROOT)}:{row_number}:{field}: value differs from schema const")
                    break
                if pattern is not None and re.fullmatch(str(pattern), value) is None:
                    findings.append(f"{csv_path.relative_to(ROOT)}:{row_number}:{field}: value differs from schema pattern")
                    break
    return findings


def _validate_normalized_provenance() -> list[str]:
    findings: list[str] = []
    dictionary_path = ROOT / "release_artifacts/provenance_data_dictionary.json"
    if not dictionary_path.is_file():
        return findings
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    if set(dictionary.get("fields") or {}) != set(PROVENANCE_FIELDS):
        findings.append("release_artifacts/provenance_data_dictionary.json: provenance field set mismatch")

    surfaces = [
        (ROOT / "release_artifacts/normalized_evidence_cards.csv", 1227, CARD_PROVENANCE),
        (ROOT / "release_artifacts/normalized_forecast_traces.csv", 336, TRACE_PROVENANCE),
    ]
    for path, expected_count, expected_profile in surfaces:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
        relative = path.relative_to(ROOT)
        missing = sorted(set(PROVENANCE_FIELDS) - fields)
        if missing:
            findings.append(f"{relative}: missing provenance fields {missing}")
        if "release_status" in fields:
            findings.append(f"{relative}: ambiguous release_status must be replaced by orthogonal provenance fields")
        if len(rows) != expected_count:
            findings.append(f"{relative}: expected {expected_count} rows, found {len(rows)}")
        for field, expected in expected_profile.items():
            counts: dict[str, int] = {}
            for row in rows:
                value = str(row.get(field) or "")
                counts[value] = counts.get(value, 0) + 1
            if counts != {expected: len(rows)}:
                findings.append(f"{relative}: invalid {field} counts {counts}")

    release_manifest_path = ROOT / "release_artifacts/release_manifest.json"
    if release_manifest_path.is_file():
        manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "came_public_release_v3":
            findings.append("release_artifacts/release_manifest.json: expected v3 release schema")
        if "source_text_policy" in manifest:
            findings.append("release_artifacts/release_manifest.json: ambiguous source_text_policy remains")
        profiles = manifest.get("artifact_provenance") or {}
        if profiles.get("normalized_evidence_cards.csv") != CARD_PROVENANCE:
            findings.append("release_artifacts/release_manifest.json: card provenance profile mismatch")
        if profiles.get("normalized_forecast_traces.csv") != TRACE_PROVENANCE:
            findings.append("release_artifacts/release_manifest.json: trace provenance profile mismatch")
        counts = manifest.get("provenance_label_counts") or {}
        if int((counts.get("normalized_evidence_cards.csv") or {}).get("rows") or -1) != 1227:
            findings.append("release_artifacts/release_manifest.json: card provenance count mismatch")
        if int((counts.get("normalized_forecast_traces.csv") or {}).get("rows") or -1) != 336:
            findings.append("release_artifacts/release_manifest.json: trace provenance count mismatch")

    replay_manifest_path = ROOT / "replay_inputs/retained_336/manifest.json"
    if replay_manifest_path.is_file():
        manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "came_retained_336_replay_inputs_v2":
            findings.append("replay_inputs/retained_336/manifest.json: expected v2 provenance schema")
        if "source_text_policy" in manifest:
            findings.append("replay_inputs/retained_336/manifest.json: ambiguous source_text_policy remains")
        if manifest.get("artifact_provenance") != REPLAY_MANIFEST_PROVENANCE:
            findings.append("replay_inputs/retained_336/manifest.json: manifest provenance profile mismatch")
        counts = manifest.get("provenance_label_counts") or {}
        expected_counts = {
            "manifest_bound_files": 401,
            "native_forward_card_rows": 1227,
            "typed_retrieval_card_rows": 2416,
        }
        if counts != expected_counts:
            findings.append(f"replay_inputs/retained_336/manifest.json: provenance counts mismatch {counts}")
    return findings


def validate_tree(root: Path = ROOT, *, archive_mode: bool = False) -> list[str]:
    findings = []
    if archive_mode:
        for path in root.rglob("*"):
            rel = path.relative_to(root)
            if any(part in BLOCKED_MEMBER_PARTS for part in rel.parts):
                findings.append(f"{rel}: blocked archive member")
            if path.is_file() and BLOCKED_NAME_RE.search(path.name) and "summary" not in path.name.lower():
                findings.append(f"{rel}: raw/private-looking archive member")
    for data_root in DATA_ROOTS:
        if not data_root.exists():
            continue
        for path in data_root.rglob("*"):
            if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                findings.extend(_scan_json_file(path))
            elif suffix == ".csv":
                findings.extend(_scan_csv_file(path))
            elif suffix == ".md":
                findings.extend(_scan_markdown_file(path))
            elif suffix == ".html":
                findings.extend(_scan_html_file(path))
    findings.extend(_scan_checkout_hygiene())
    findings.extend(_validate_required_surfaces())
    findings.extend(_validate_complete_schemas())
    findings.extend(_validate_normalized_provenance())
    return findings


def main() -> None:
    findings = validate_tree()
    if findings:
        print("quote-free release validation failed:", file=sys.stderr)
        for finding in findings[:100]:
            print(f"- {finding}", file=sys.stderr)
        if len(findings) > 100:
            print(f"- ... {len(findings) - 100} additional findings", file=sys.stderr)
        raise SystemExit(1)
    print("quote-free release validation passed")


if __name__ == "__main__":
    main()
