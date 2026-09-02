# Harmonized Call100 Candidate Evaluation

The raw and controlled AMT conditions reuse the exact A0 and A6 batches from the module ablation. The motif batch is retained from the rule-based run, and all 27,000 MIDI files are re-scored with the same versioned evaluator.

- Score version: `structural_compliance_v1.1`
- Rows: `27000`
- Maximum score drift after re-scoring: `0.000000500`

## Candidate Summary

| candidate | n | composite | style compliance | non-style structural |
| --- | ---: | ---: | ---: | ---: |
| amt_small_controlled | 9000 | 0.732610 | 0.809814 | 0.676704 |
| amt_small_raw | 9000 | 0.560968 | 0.563724 | 0.558971 |
| motif_transform_baseline | 9000 | 0.752682 | 0.854195 | 0.679172 |

## Call-clustered comparisons

The 9,000 repeated rows per candidate are descriptive, not independent musical inputs. Primary uncertainty first averages the shared preset/trial rows within each of 100 Calls; source sensitivity uses 87 equally weighted source clusters. The authoritative generated tables and full provenance are in `results/paper_clustered_statistics/`.

| comparison | mean difference | Call 95% CI | positive Calls | source 95% CI |
| --- | ---: | ---: | ---: | ---: |
| controlled_minus_raw | 0.171642 | [0.157901, 0.185151] | 100/100 | [0.158728, 0.187555] |
| controlled_minus_motif | -0.020072 | [-0.031425, -0.008749] | 40/100 | [-0.032944, -0.008721] |
| motif_minus_raw | 0.191714 | [0.174714, 0.208189] | 96/100 | [0.176257, 0.211206] |
