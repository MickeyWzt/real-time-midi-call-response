# Zenodo DOI Workflow

This repository is prepared for Zenodo-GitHub archiving.

## Metadata Files

- `CITATION.cff` lets GitHub display citation metadata.
- `.zenodo.json` gives Zenodo software-specific metadata for release archiving.

Zenodo gives priority to `.zenodo.json` when both files exist.

## Steps

1. Make the GitHub repository public.
2. Log in to Zenodo with the same GitHub account.
3. Open the Zenodo GitHub page and click **Sync now**.
4. Toggle this repository on.
5. Create a GitHub release, for example `v1.0.0`.
6. Wait for Zenodo to archive the release and mint a version DOI.
7. Update the README and GitHub Pages DOI badge after the DOI appears.

GitHub release archiving creates a new Zenodo DOI for each release. Use the version DOI for exact reproducibility and the concept DOI for the project as a whole.
