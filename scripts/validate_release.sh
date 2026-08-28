#!/usr/bin/env bash
set -euo pipefail

bash scripts/validate_package.sh
python scripts/replay_saved_artifacts.py
bash scripts/run_reference_replay.sh
