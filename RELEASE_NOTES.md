# v1.2.0

Evidence-semantics and reproducibility release for **Adapting a Frozen Anticipatory Music Transformer for Turn-Based MIDI Call-and-Response**.

All-versions Zenodo concept DOI: https://doi.org/10.5281/zenodo.20838083

The exact v1.2.0 version DOI is recorded on the GitHub release page after Zenodo archival. The prior v1.1.0 DOI is `10.5281/zenodo.21860065` and must not be used to identify this revision.

## Major corrections

- Replaces post-hoc endpoint scoring with a first-commit replay that terminates after the deployed state machine's first Commit.
- Defines the custom premature-sensitive first-commit score (PS-F1): `TP=success`, `FP=premature`, and `FN=premature+late+missed`; it is not conventional event-detection F1.
- Reinterprets the legacy `first_token_latency` field as time to the first complete generated event and reports a commit-to-first-generated-event readiness lower bound, not MIDI transmission or audio-onset latency.
- Prevents the A4 motif-fallback path from silently applying A5/A6 style or theory controls.
- Replaces trial-row inference with 100-Call bootstrap inference and an 87-source-cluster sensitivity analysis for candidate, ablation, and scheduler results.
- Records both the original endpoint input-manifest hash and the published path-redacted manifest hash so the provenance chain can be verified without exposing local paths.

## New evidence and tooling

- 27,000-row matched-versus-mismatched symbolic feature-association audit with bank-rebuilding Call/source jackknife analyses.
- Cluster-aware candidate, stepwise-ablation, and scheduler-replay reports with fixed seeds and input hashes.
- Expanded endpoint sensitivity conditions for clustering, confirmation, threshold, window, intensity floor, and cutoff clamps.
- Runtime revalidation of the 27,000-row structural score under Python 3.12.10 and zlib 1.3.1.
- Additional regression tests for endpoint first-commit semantics, A4 fallback isolation, and feature-association calculations.
- Revised 29-page manuscript, figures, README, project page, metadata, and release documentation.

## Human-evidence boundary

No human-participant data or aggregate outcomes are included as evidence in v1.2.0. The earlier formative listening exercise lacked prospective institutional approval or exemption and did not enforce adult eligibility. Its aggregate output files have been removed from this release; historical v1.1.0 artifacts remain only in that immutable prior tag and must not be interpreted as an ethically governed human-subject result.

Future listening or live-performance evaluation must begin only after prospective institutional review or a documented determination that review is not required, with age eligibility, consent, withdrawal, retention, and de-identification procedures fixed before recruitment.

## Claim boundary

This release supports an inspectable engineering adaptation, structural manipulation checks, narrow symbolic feature associations, and deterministic scheduler behavior. It does not establish musical quality, universal endpoint accuracy, faster intrinsic AMT decoding, MIDI/audio latency, human-responsive turn taking, or perceptual superiority.

## Excluded from the archive

- model weights and third-party audio software
- license-sensitive source datasets and generated MIDI response batches
- all human-participant response data and aggregate outcomes
- private answer keys and deployment credentials
- piano sample libraries, VST plugins, and DAWs

Published result tables use path-redacted filenames and SHA-256 values. Recomputing the feature-association analysis from MIDI requires locally regenerated responses; this limitation is stated in the manuscript and metadata.
