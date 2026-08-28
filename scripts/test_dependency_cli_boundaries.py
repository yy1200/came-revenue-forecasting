#!/usr/bin/env python3
"""Verify that implementation dependencies cannot be mistaken for public CLIs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "company_stat_anchor_shock.run_csais_v1",
    "company_stat_anchor_shock.run_csais_candidate_bridge_v1",
    "company_stat_anchor_shock.run_csais_rawcard_direct_experts_v1",
    "native_evidence_forecaster.run_backbone_v2",
    "native_evidence_forecaster.run_native_cards_v1",
    "native_evidence_forecaster.run_native_csais_v1",
)
def main() -> None:
    findings: list[str] = []
    env = os.environ.copy()
    env.pop("CAME_REFERENCE_REPLAY", None)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for module in MODULES:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0 or "Implementation dependency only" not in output:
            findings.append(f"{module}: standalone CLI boundary failed")
    if findings:
        raise SystemExit("\n".join(findings))
    print(f"dependency CLI boundary validation passed (modules={len(MODULES)})")


if __name__ == "__main__":
    main()
