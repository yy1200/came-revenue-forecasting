# 336-Row Reference Replay Inputs

This compatibility-named directory contains quote-free post-extraction inputs for deterministic execution of the CAME runner over all `336` reference company-quarter rows.

`manifest.json` applies five independent provenance dimensions to the manifest and replay-input family: public inclusion, derived-artifact type, source-material access, source-material redistribution, and verbatim-text status. The values are defined in `release_artifacts/provenance_data_dictionary.json`. They do not alter file payloads, runner behavior, predictions, or replay calculations.

The replay inputs are not saved final predictions. They contain structured forecast and attribution features, author-generated statistical anchor candidates, normalized forward evidence cards, typed retrieval cards, and the frozen precomputed no-guidance proposal required by anchor memory. Direct third-party source text and raw corpora are excluded. Opaque per-release token IDs preserve token-set overlap without publishing token identities or the unreleased HMAC key.

`ram_proposal_input.csv` is the only precomputed prediction-like CAME input. Its compatibility field is the frozen upstream no-guidance proposal consumed by anchor memory. It is not the final CAME prediction: the runner independently computes the pre-memory prediction, applies prior-only admission, and uses the proposal only on admitted no-guidance rows. Exact compatibility names are defined in `GLOSSARY.md`.

The released artifact has a documented two-stage AAPL correction boundary. The replay first runs all `336` rows with the pre-correction AAPL `FY2019_Q4` statistical guidance candidate and anchor-memory proposal, then runs corrected AAPL inputs through the same runner and takes only the corrected `FY2019_Q4` row. No saved correction-row prediction is supplied.

The historical runner output clipped `demand_mentions` and `supply_constraint_mentions` to `30` in its emitted diagnostics even though the model consumed the original integer counts. Release-input construction recovers those two post-extraction counts by inverting the emitted `factor_demand` and `factor_supply` equations and fails unless both recovered values are integers. This reconstructs features, not predictions.

Run:

```bash
python scripts/validate_reference_inputs.py
bash scripts/run_reference_replay.sh
```

The replay writes generated runner artifacts and the final comparison under marker-owned, ignored `output/reference_replay/`. It refuses to remove an existing directory without the exact ownership marker.

Input-construction and pseudonymization utilities are intentionally excluded. Ordinary public replay uses only the checked-in manifest-bound inputs and does not require excluded data or the unreleased HMAC key.
