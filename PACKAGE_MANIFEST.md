# Public Package Manifest

## Purpose

This manifest defines the minimal public release for **CAME: Company-Aware Evidence-Memory Experts**. The canonical URL is <https://github.com/yy1200/came-revenue-forecasting>.

The package is a quote-free post-extraction reproducibility surface, not the full author workspace or an upstream data-engineering release.

## Included

### Compatibility Runner

- `company_stat_anchor_shock/run_csais_rawcard_direct_experts_v1.py`: CAME compatibility runner.
- `company_stat_anchor_shock/run_csais_v1.py`: anchor, feature, metric, and retrieval helpers imported by the runner.
- `company_stat_anchor_shock/run_csais_candidate_bridge_v1.py`: intrinsic/current-card and compressed-candidate helpers imported by the runner.
- `evidence_memory_residual/`, selected `native_evidence_forecaster/*.py`, and `temporal_kg_memory_attention/pair_features.py`: exact imported utility closure.

Historical filenames are retained for executable fidelity and defined in `GLOSSARY.md`. Only `scripts/run_reference_replay.sh` defines the supported full-run invocation; dependency modules reject standalone execution.

### Contract And Schemas

- `release_artifacts/came_release_contract.json`: method identity, panel, split, anchor rule, compatibility flags, inactive paths, and replay boundary.
- `replay_inputs/retained_336/experiment.json` and `companies/*/company.json`: self-contained public replay configuration.
- `release_artifacts/schemas/`: normalized card and trace schemas.
- `prompt_specs/schemas/`: direct-forecast and evidence-extraction response schemas.

### Prompt Specifications

- `prompt_specs/executable/direct_same_task_forecast.py`: exact identity-masked History-only and History + Guidance prompt/parser contract.
- `prompt_specs/executable/came_evidence_card_extraction.py`: exact strict target-quarter causal-driver relation/event request contract.
- `prompt_specs/spec_manifest.json`: model, parser, request, source-hash, file-hash, and canonical-render metadata.

### Normalized Public Artifacts

- `release_artifacts/normalized_evidence_cards.csv`: `1,227` structured quote-free cards.
- `release_artifacts/normalized_forecast_traces.csv`: all `336` final CAME predictions, online anchors, execution-lineage traces, card references, and provenance labels.
- `release_artifacts/provenance_data_dictionary.json`: definitions for public inclusion, derived-artifact type, source access, source redistribution, and verbatim-text status.
- `release_artifacts/release_manifest.json`: public artifact hashes, counts, and release boundary.

### Full Replay Inputs

- `replay_inputs/retained_336/manifest.json`: hashes and roles for `401` replay files.
- `replay_inputs/retained_336/companies/`: structured forecast, attribution, statistical-anchor, and company inputs for all `12` companies.
- `replay_inputs/retained_336/normalized_native_forward_cards.jsonl`: `1,227` quote-free forward cards with opaque token IDs.
- `replay_inputs/retained_336/retrieve/`: `336` typed-retrieval payloads containing `2,416` structured context cards.
- `replay_inputs/retained_336/ram_proposal_input.csv`: frozen upstream no-guidance proposal required by anchor memory; not a final CAME prediction.

### Public Commands

- `scripts/run_reference_replay.sh` and `scripts/replay_reference.py`: full reference replay.
- `scripts/replay_saved_artifacts.py`: normalized trace and headline metric replay.
- `scripts/validate_release_contract.py`: executable contract alignment.
- `scripts/validate_reference_inputs.py`: input integrity and no-final-prediction gate.
- `scripts/validate_prompt_specs.py`: prompt/schema/render validation.
- `scripts/validate_quote_free_release.py`: privacy, provenance, card-link, and release-surface validation.
- `scripts/validate_manifest_hashes.py`: manifest SHA-256 validation.
- `scripts/test_replay_output_safety.py`: output containment and ownership safety.
- `scripts/test_dependency_cli_boundaries.py`: dependency-module CLI boundary checks.
- `scripts/validate_package.sh` and `scripts/validate_release.sh`: package and full replay gates.

## Explicitly Excluded

- raw transcripts, filings, vendor corpora, direct source excerpts, and quote-bearing traces;
- upstream API requests/responses, prompt caches, credentials, and environment files;
- private source acquisition, extraction, grounding, KG rebuild, and statistical-training pipelines;
- replay-input construction, sanitization, overlap checking, and prebuilt reconstruction tools;
- historical output trees, unused KG interfaces, old fusion/external paths, and exploratory source-policy analyses;
- standalone dependency-module command-line interfaces and unsupported evaluators;
- draft company configurations and unreleased workflow state;
- direct-comparator row-level outputs and standalone third-party model outputs; and
- generated output trees, notebooks, caches, archives, and Git history.

The public conversion changes no final prediction, method decision, comparator contract, evaluation row, or CAME/anchor metric.
