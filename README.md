# CAME Revenue Forecasting

Code and reproducibility package for **CAME: Company-Aware Evidence-Memory Experts for Interpretable Quarter-Ahead Revenue Forecasting**.

Canonical repository: <https://github.com/yy1200/came-revenue-forecasting>

Authors: Ya-Wen Wu, Meng-Fen Chiang, Kuang-Da Wang, and Wen-Chih Peng, National Yang Ming Chiao Tung University.

## Release Boundary

This is a minimal quote-free post-extraction release. It contains the CAME compatibility runner and its required dependencies, exact release settings, all `336` normalized predictions and traces, `1,227` normalized evidence cards, manifest-bound replay inputs, executable prompt/schema specifications, validation scripts, and licensing/provenance metadata.

It does not redistribute transcripts, filings, vendor corpora, direct source excerpts, upstream API responses, prompt caches, standalone third-party model outputs, source-acquisition tools, or source-grounding pipelines.

## Supported Entry Point

Run the full reference replay through:

```bash
bash scripts/run_reference_replay.sh
```

The implementation modules retain machine identifiers required by deterministic replay, but reject standalone execution. `GLOSSARY.md` defines those identifiers and separates them from public CAME terminology. The exact supported settings, split, panel, anchor rule, inactive paths, and replay boundary are recorded in `release_artifacts/came_release_contract.json` and checked against the executable wrapper.

The release contract is online anchor/control plus guidance-gated anchor memory, partitioned current-evidence residuals, typed temporal memory, and strict explicit-guidance residuals. External experts, KG forecast deltas, historical fusion selection, action gates, and the segment bridge are not active forecast paths.

## Reproduction

1. Full replay: `bash scripts/run_reference_replay.sh` regenerates all `336` forecasts from quote-free post-extraction inputs and compares them with `release_artifacts/normalized_forecast_traces.csv`.
2. Metric replay: `python scripts/replay_saved_artifacts.py` validates the normalized trace surface and recomputes development-inclusive and temporal held-out CAME/anchor macro sMAPE and pooled MAE.
3. Prompt validation: `python scripts/validate_prompt_specs.py` checks the exact identity-masked History-only and History + Guidance request specifications and the strict target-quarter causal-driver extraction contract without making API calls.

These modes do not reproduce source acquisition, document extraction, source-grounding judgments, or excluded preprocessing. The full replay uses one documented precomputed prediction-like no-guidance proposal, but supplies no saved final CAME prediction. It also reproduces the historical two-stage AAPL `FY2019_Q4` input correction described in `GLOSSARY.md`.

## Recomputed Results

| Surface | N | CAME macro sMAPE | Online anchor macro sMAPE | CAME pooled MAE | Online anchor pooled MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development-inclusive all-12 | `336` | `0.059253` | `0.069950` | `1.810B` | `2.251B` |
| Temporal held-out all-12 | `96` | `0.037810` | `0.047171` | `1.483B` | `1.929B` |

MAE magnitudes use each company's reporting currency, USD except ASML in EUR, without FX normalization. Method and protocol selection used only AAPL, NVDA, and AVGO through `FY2023_Q4`; all settings were fixed before the `FY2024_Q1` to `FY2025_Q4` temporal held-out and nine-company-held-out evaluations.

The exact direct same-task prompt contracts are included, but direct-comparator API responses, caches, and row-level outputs are intentionally excluded. This repository therefore does not claim to recompute direct-comparator metrics.

## Validate

Install the minimal dependencies and run the complete release gate:

```bash
pip install -r requirements.txt
bash scripts/validate_release.sh
```

The gate validates syntax, JSON, prompt and artifact hashes, quote-free/provenance constraints, complete normalized schemas, the release contract, replay-output safety, normalized metrics, and the full `336/336` runner replay.

## Licensing

- Code: Apache License 2.0, see `LICENSE`.
- Author-created normalized artifacts: CC BY 4.0, subject to `DATA_LICENSE.md`.
- Citation metadata: `CITATION.cff`.

The artifact license does not relicense third-party source text, filings, transcripts, vendor data, API responses, standalone third-party model outputs, or trademarks.

## Read First

1. `REPRODUCE.md`
2. `GLOSSARY.md`
3. `PACKAGE_MANIFEST.md`
4. `release_artifacts/came_release_contract.json`
5. `release_artifacts/README.md`
6. `replay_inputs/retained_336/README.md`
7. `prompt_specs/README.md`
8. `DATA_LICENSE.md`
