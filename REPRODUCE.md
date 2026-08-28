# Reproduce And Validate

This package supports deterministic post-extraction replay of the frozen CAME forecast path. It excludes upstream source acquisition, source text, closed-model responses, extraction execution, and source-grounding judgments.

## 1. Install

```bash
pip install -r requirements.txt
```

The runtime dependencies are NumPy and pandas. Validation otherwise uses the Python standard library and common Unix tools.

## 2. Run The Complete Gate

```bash
bash scripts/validate_release.sh
```

This command:

- compiles the included Python files and parses all JSON;
- imports the CAME compatibility-runner dependency closure;
- checks quote-free, provenance, identifier, secret, and release-surface constraints;
- validates replay-input, prompt-spec, and normalized-artifact hashes;
- checks the fixed compatibility flags, panel, split, anchor settings, and replay boundary;
- tests replay-output ownership and symlink safety;
- recomputes saved CAME/anchor headline metrics; and
- regenerates and matches all `336` reference forecasts.

## 3. Replay All 336 Forecasts

```bash
bash scripts/run_reference_replay.sh
```

The wrapper validates the public contract and all `401` manifest-bound replay files, then executes the compatibility runner over `12` companies and `28` quarters per company. Expected status:

```text
status=pass rows=336 companies=12 matched=336 mismatched=0 runner_generated=336 final_prediction_inputs=0
```

The replay first regenerates the all-`336` pre-correction candidate, reruns corrected AAPL inputs, and uses only the runner-generated corrected `FY2019_Q4` row. `ram_proposal_input.csv` contains the frozen upstream no-guidance proposal required by the compatibility runner; it is prediction-like but is not a final CAME prediction. See `GLOSSARY.md` for exact machine-field meanings.

Generated artifacts are written under marker-owned, ignored `output/reference_replay/`. The script refuses to remove an existing directory without the exact ownership marker.

## 4. Recompute Metrics

```bash
python scripts/replay_saved_artifacts.py
```

This validates the `336` unique ticker-quarter rows in `release_artifacts/normalized_forecast_traces.csv` and recomputes company-equal macro sMAPE and pooled MAE for CAME and the matched online anchor over:

- the development-inclusive `336`-row surface; and
- the `96`-row `FY2024_Q1` to `FY2025_Q4` temporal held-out surface.

The normalized trace is the public saved-prediction authority. Full replay independently compares every runner-generated final prediction against it.

## 5. Validate Prompt Specifications

```bash
python scripts/validate_prompt_specs.py
```

This checks executable prompt builders, response schemas, canonical render hashes, company/currency/target rendering, target-quarter parser rejection, and the strict target-quarter relation/event request contract. It makes no API calls and uses synthetic placeholders only.

Direct-comparator row-level outputs are excluded, so this package validates the exact prompt/parser contract but does not recompute direct-comparator metrics.

## 6. Inspect The Public Contract

`release_artifacts/came_release_contract.json` records:

- the CAME method identity and compatibility implementation alias;
- the supported launcher and prediction columns;
- the all-12 panel and reporting-currency rule;
- development, temporal held-out, and company-held-out splits;
- all fixed compatibility flags and anchor settings;
- the two-stage AAPL repair boundary; and
- explicitly inactive forecast paths.

`python scripts/validate_release_contract.py` checks this declaration against the executable replay command. `GLOSSARY.md` defines every compatibility identifier exposed by the contract and normalized artifacts.
