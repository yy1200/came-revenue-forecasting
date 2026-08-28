# Quote-Free Release Artifacts

This directory is the compact public result and contract surface for frozen CAME.

- `came_release_contract.json` records the supported method identity, compatibility flags, panel, split, anchor settings, replay tolerances, and inactive forecast paths.
- `normalized_evidence_cards.csv` contains `1,227` structured quote-free card rows with source hashes/locators and five orthogonal provenance fields.
- `normalized_forecast_traces.csv` is the public saved-prediction authority for all `336` reference rows. It contains final CAME predictions, online anchors, execution-lineage traces, and normalized card references.
- `provenance_data_dictionary.json` defines public inclusion, derived-artifact type, source access, source redistribution, and verbatim-text values.
- `release_manifest.json` records coverage, hashes, provenance counts, the public URL, and the release boundary.
- `schemas/` completely defines all `37` normalized-card and `34` forecast-trace fields. `GLOSSARY.md` explains compatibility labels and categorical values.

Run `python scripts/replay_saved_artifacts.py` to validate the trace surface and recompute development-inclusive and temporal held-out CAME/anchor headline metrics. Run `bash scripts/run_reference_replay.sh` to regenerate and compare all `336` forecasts against the normalized trace.

Neither command reproduces upstream source acquisition, closed-model extraction, source-grounding decisions, or excluded direct-comparator outputs.
