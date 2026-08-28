#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q .

python - <<'PY'
import json
from pathlib import Path

count = 0
for path in Path('.').rglob('*.json'):
    if '.git' in path.parts:
        continue
    with path.open(encoding='utf-8') as f:
        json.load(f)
    count += 1
print(f'validated {count} JSON files')
PY

python - <<'PY'
import company_stat_anchor_shock.run_csais_rawcard_direct_experts_v1 as came
print('loaded', came.__name__)
PY

python scripts/validate_quote_free_release.py
python scripts/validate_reference_inputs.py
python scripts/validate_release_contract.py
python scripts/validate_prompt_specs.py
python scripts/validate_manifest_hashes.py
python scripts/test_replay_output_safety.py
python scripts/test_dependency_cli_boundaries.py
