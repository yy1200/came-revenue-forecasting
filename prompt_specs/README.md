# Prompt And Schema Specifications

This package documents the prompt contracts used by the CAME release without publishing API responses or third-party source text.

## Exact Executable Contracts

- `executable/direct_same_task_forecast.py` reproduces the complete identity-masked History-only and History + Guidance system/user prompt builders, strict JSON response format, and parser checks used for the reported direct comparator completed on 2026-08-27.
- `executable/came_evidence_card_extraction.py` reproduces the strict target-quarter relation and causal-driver event system/user messages, JSON-safe text truncation, and deterministic length-fallback request. The relation request omits the temperature parameter as in the source call; primary and fallback event requests use `temperature=0.0`.
- `schemas/` contains the direct-forecast response schema and the Pydantic-level claim/event output schemas consumed by the extraction parser.
- `spec_manifest.json` records models, parser/runtime versions where recorded, unavailable source-artifact hashes, and hashes for every included file.

The executable builders accept placeholders or caller-supplied CSV/document text at runtime. Canonical tests use synthetic placeholders only. No company data, rendered source request, source excerpt, model response, credential, or cache is included.

## Boundaries

- Direct forecasting replaces the company-name token with `Entity X`, retains exact fiscal periods, reporting currency, and absolute scale, uses expanding prior-quarter history, and enforces exact target-quarter validation.
- History + Guidance falls back to the History-only result when target-quarter numeric guidance is unavailable; no second request is made when the History result is already available.
- Evidence extraction is upstream of the public post-extraction replay. The release provides the prompt/schema contract but not source acquisition, source text, API responses, or a claim that the normalized cards can be regenerated without the excluded source material.
- The extraction request specifications identify the original Pydantic response-model names. They do not call an API or claim to export the separate downstream grounding and deduplication implementation.
- Provider model aliases are reported exactly as recorded. A snapshot is not inferred when the original extraction artifacts recorded only an alias.
- Legacy policy aliases used by the frozen source lineage are documented in `GLOSSARY.md`; they are not public method names.

Validate this package with `python scripts/validate_prompt_specs.py`.
