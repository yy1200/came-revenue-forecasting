# Data And Artifact Licensing

## Author-Created Artifacts

To the extent the CAME authors hold the relevant rights, the author-created normalized artifacts in `release_artifacts/` and the quote-free derived replay artifacts in `replay_inputs/retained_336/` are licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/), also identified as `CC-BY-4.0`.

Attribution should identify the CAME authors, the artifact title, the repository URL, and the artifact version or commit used. Citation metadata is in `CITATION.cff`.

## Excluded Rights

This license does not grant rights in:

- third-party filings, earnings-call transcripts, vendor datasets, API responses, or source excerpts;
- third-party verbatim text represented only by hashes, opaque token IDs, source IDs, dates, or locators;
- standalone outputs of third-party models or rights retained by a model provider;
- company names, trademarks, product names, or linked third-party websites; or
- any permission-dependent quote-bearing trace not included in this release.

The CC BY 4.0 grant covers the authors' selection, arrangement, normalization, annotations, trace structure, and numerical evaluation artifacts only to the extent those elements are copyrightable and controlled by the authors. It does not purport to relicense underlying third-party material or resolve ownership questions for model-generated content.

## Release Boundary

The public release contains quote-free normalized cards/traces and manifest-bound post-extraction inputs for deterministic all-`336` forecast replay. It does not reproduce source acquisition, source-grounding judgments, or upstream extraction. No permission-dependent text is included.

Source hashes and opaque token IDs are integrity/replay metadata, not substitutes for source licenses and not a grant of access to private corpora. Raw transcript, filing, vendor, and upstream API corpora are not redistributed.

`release_artifacts/provenance_data_dictionary.json` defines five independent labels used by normalized tables and replay manifests: public release status, derived-artifact status, source-material access status, source-material redistribution status, and verbatim-text status.

## Code

Software source, scripts, prompt templates, schemas, and configuration files are licensed under Apache License 2.0 unless a file states otherwise. See `LICENSE` and `NOTICE`.
