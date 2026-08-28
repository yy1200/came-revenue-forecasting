#!/usr/bin/env python3
"""Validate SHA-256 records in public package manifests using the standard library."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MANIFESTS = (
    (ROOT / "release_artifacts/release_manifest.json", ROOT / "release_artifacts"),
    (ROOT / "replay_inputs/retained_336/manifest.json", ROOT / "replay_inputs/retained_336"),
    (ROOT / "prompt_specs/spec_manifest.json", ROOT / "prompt_specs"),
)


def _record_hash(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return str(record.get("sha256") or "")
    return ""


def validate() -> tuple[list[str], int]:
    findings: list[str] = []
    checked = 0
    for manifest_path, artifact_root in MANIFESTS:
        manifest_rel = manifest_path.relative_to(ROOT)
        if not manifest_path.is_file():
            findings.append(f"{manifest_rel}: missing manifest")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(f"{manifest_rel}: invalid manifest: {exc}")
            continue
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            findings.append(f"{manifest_rel}: missing non-empty files map")
            continue
        for raw_relative, record in files.items():
            relative = Path(str(raw_relative))
            if relative.is_absolute() or ".." in relative.parts:
                findings.append(f"{manifest_rel}: unsafe manifest path {raw_relative!r}")
                continue
            path = artifact_root / relative
            expected = _record_hash(record)
            if not HEX64.fullmatch(expected):
                findings.append(f"{manifest_rel}: invalid SHA-256 for {raw_relative}")
                continue
            if not path.is_file():
                findings.append(f"{manifest_rel}: missing file {raw_relative}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                findings.append(f"{manifest_rel}: SHA-256 mismatch for {raw_relative}")
            if isinstance(record, dict) and "bytes" in record and path.stat().st_size != record["bytes"]:
                findings.append(f"{manifest_rel}: byte-size mismatch for {raw_relative}")
            checked += 1
    return findings, checked


def main() -> None:
    findings, checked = validate()
    if findings:
        print("manifest hash validation failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        raise SystemExit(1)
    print(f"manifest hash validation passed (files={checked} manifests={len(MANIFESTS)})")


if __name__ == "__main__":
    main()
