# Public Terminology And Compatibility Glossary

This glossary separates the published CAME concepts from frozen execution-lineage identifiers that remain only because deterministic replay depends on them. Compatibility identifiers are not additional methods, paper claims, or recommended entry points.

## Supported Public Terms

| Term | Meaning in this release |
| --- | --- |
| CAME | Company-Aware Evidence-Memory Experts, the released quarter-ahead revenue forecasting method. |
| Online anchor/control | The matched prior-only numerical forecast selected online from information available before the target quarter. |
| Anchor memory | The guidance-gated, prior-only mechanism that may admit an upstream no-guidance proposal relative to the online anchor. Historical code and trace fields abbreviate this mechanism as `RAM`. |
| Current-evidence residual | The partitioned intrinsic residual derived from normalized evidence cards available for the forecast origin. Historical execution fields use `intrinsic`. |
| Typed temporal memory | Prior evidence-card context retrieved under the released temporal-memory contract. Historical execution fields use `temporal`. |
| Explicit-guidance residual | The residual channel active only under the released explicit-guidance contract. Historical execution fields use `guidance`. |
| Strict target-quarter causal-driver policy | Event extraction accepts only causal drivers explicitly tied to the target quarter. The frozen source-lineage alias is `u0_strict_driver`. |
| Reference prediction surface | The `336` ticker-quarter final CAME predictions in `release_artifacts/normalized_forecast_traces.csv`. |

The conceptual method should be read from the paper and `release_artifacts/came_release_contract.json`. Execution-lineage fields expose enough frozen state to reproduce the released numbers; they are not a one-to-one renaming of paper components and should not be used to infer a different architecture.

## Machine Compatibility Identifiers

| Identifier | Public interpretation |
| --- | --- |
| `came_ram_v1p1_partitioned_residual_retained_20260511` | Frozen implementation alias for this release. `retained` and the date are lineage tokens, not method names. |
| `company_stat_anchor_shock.run_csais_rawcard_direct_experts_v1` | Python module imported by the supported replay. `CSAIS`, `rawcard`, and `v1` are historical code tokens, not separately released methods. |
| `replay_inputs/retained_336` | Compatibility directory containing the manifest-bound inputs for the `336`-row reference surface. |
| `pred_csais_rawcard_direct_experts_v1` | Frozen runner output column mapped to public `final_came_prediction`. |
| `pred_csais_anchor` | Frozen runner output column mapped to public `online_anchor_prediction`. |
| `pred_came_ge_v1p1_noguidance_anchor_guard_candidate` | Precomputed upstream no-guidance proposal consumed by anchor memory. It is prediction-like but is not a saved or final CAME prediction. |
| `CAME-GE` / `came_ge` | Historical label attached only to the upstream proposal above; it is not an active external expert or a separately reported method. |
| `signed_strength_v1` | Frozen prior-only admission rule for the upstream proposal on eligible no-guidance rows. |
| `main_all12_topall` | Compatibility value selecting the proposal rows that correspond to the released all-12 surface. |
| `full_mainpanel` | Card-lineage value for the full available-quarter card construction window. It is not a claim that every card belongs to a held-out evaluation split. |
| `reduced_window` | Card-lineage value for a shorter available-quarter construction window. |
| `ledger_forward` | Card assembled from the forward relation/claim stream. |
| `shock_forward` | Card also supported by the strict target-quarter event stream. The word `shock` is a historical extraction-stream label, not a causal-estimation claim. |
| `ledger_forward+shock_forward` | Card present in both forward extraction streams. |
| `conservative_admissibility_v1` / `conservative_admissibility_v2` | Historical helper-mode identifiers for deterministic card weighting. They are implementation dependencies, not reported alternatives. |

Only `bash scripts/run_reference_replay.sh` is the supported full-run entry point. Dependency modules reject standalone execution because their historical defaults do not define the released CAME configuration.

## Forecast Trace Fields

The complete machine-readable field contract is `release_artifacts/schemas/normalized_forecast_trace.schema.json`.

| Field or value | Meaning |
| --- | --- |
| `actual` | Reported target-quarter revenue in the company's reporting currency. Values are not FX-normalized. |
| `baseline_prediction` | Compatibility numeric baseline field from the shared forecast input table. It is not the identity-masked direct same-task LLM comparator. |
| `online_anchor_prediction` | Matched online anchor/control prediction. |
| `pre_ram_prediction` | CAME prediction immediately before the anchor-memory admission step. It is not the final prediction when `ram_active=1`. |
| `final_came_prediction` | Public final CAME prediction and reference value for deterministic replay. |
| `compressed_base_*` | Historical execution-lineage channel used before residual combination; it is not an additional published expert. |
| `*_delta_log` | Additive correction on the natural-log revenue scale for the named execution channel. |
| `*_support` | Nonnegative support assigned to the named channel before final combination. |
| `*_weight` | Effective blend weight recorded for the named execution channel. |
| `blend_support_sum` | Sum of the recorded effective channel supports used in the blend trace. |
| `guidance_active` | `1` when the explicit-guidance residual channel is active, otherwise `0`. |
| `ram_active` | `1` when anchor memory admits the upstream proposal, otherwise `0`. |
| `ram_history_n` | Number of prior eligible no-guidance observations available to the admission rule. |
| `ram_prior_signed_log_error` | Prior-only signed log-error statistic used by the admission rule; blank when unavailable. |
| `normalized_card_ids_json` | JSON array of card IDs linked to the forecast trace. |

`ram_reason` values are exact compatibility outputs:

| Value | Meaning |
| --- | --- |
| `not_no_guidance_current_kept` | The row is not eligible for the no-guidance proposal; the pre-memory prediction is kept. |
| `insufficient_prior_no_guidance_history_current_kept` | Eligible row lacks the required prior no-guidance history; the pre-memory prediction is kept. |
| `proposal_not_upward_current_kept` | The proposal does not satisfy the frozen upward-proposal condition; the pre-memory prediction is kept. |
| `prior_signed_error_strength_active` | Prior-only signed error exceeds the frozen admission threshold and the proposal is admitted. |

`guidance_availability` values distinguish `explicit_numeric`, `derived_weak_numeric`, `qualitative_only`, `no_total_revenue_guidance_but_forward_commentary`, and `none`. Only the released runner contract determines whether the guidance residual is active.

## Evidence Card Fields

The complete machine-readable field contract is `release_artifacts/schemas/normalized_evidence_card.schema.json`.

| Field or value | Meaning |
| --- | --- |
| `card_id` | Release-local stable identifier for a normalized card. |
| `instance_id`, `signature_key`, `attribution_anchor` | Opaque equality-preserving 24-character identifiers. They reveal no original identifier without the unreleased key. |
| `driver_source=claim` | Card originated from the relation/claim extraction stream. |
| `driver_source=event` | Card originated from the strict target-quarter causal-driver extraction stream. |
| `support_type=explicit` | Target attribution is directly represented in the extracted structure. |
| `support_type=inferred` | Attribution was derived by the post-extraction structured mapping; this is not a human-verified causal claim. |
| `support_type` blank | No support-type label was available. |
| `attribution_method=explicit_segment` | Explicit segment attribution. |
| `attribution_method=explicit_segment_with_path` | Explicit segment attribution with a structured relation path. |
| `attribution_method=explicit_tail` | Attribution uses an explicitly extracted tail entity. |
| `attribution_method=unresolved` | No supported segment-level attribution was assigned. |
| `panel_type` | Card-construction coverage label, defined in the compatibility table above; not an evaluation-split label. |
| `archetype` | Shared company-configuration category used by the unified pipeline, not a per-company method branch. |
| `source_sentence_ids_json` | JSON array of source-local sentence locators. The source sentences are not redistributed. |
| `source_text_sha256` | Unsalted SHA-256 of the withheld supporting text. It supports release-maintainer equality checks but can confirm a guessed candidate string; it is not anonymization or a rights grant. |
| `source_text_char_count` | Character count of the withheld supporting text. |
| `is_forward_target_quarter` | Whether the normalized card was marked as forward evidence for the target quarter. |
| `is_observed_realized` | Whether the normalized card was marked as realized evidence for the observed quarter. |

`temporal_type` uses the exact extraction labels `Forecast/Guidance`, `Condition/External`, and `Realized/Reporting`. These labels describe evidence timing, not certainty or causal validity.

## Replay Boundaries

- The replay is post-extraction. It does not reacquire source documents, rerun closed-model extraction, or repeat source-grounding judgments.
- `ram_proposal_input.csv` is the only precomputed prediction-like input. The release contains zero saved final-prediction inputs.
- The historical artifact uses a documented two-stage correction for AAPL `FY2019_Q4`: all rows are regenerated once with the pre-correction input, AAPL is regenerated with the corrected input, and only that corrected row is selected.
- The normalized cards and traces contain no verbatim third-party source text. Source hashes and locators do not grant source access or redistribution rights.
- The evidence-extraction artifacts recorded the provider alias `gpt-4.1-mini` but did not uniformly record a served snapshot. This release does not infer one.
