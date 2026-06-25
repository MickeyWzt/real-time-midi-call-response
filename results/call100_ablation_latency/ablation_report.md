# Call100 Ablation Study

## Reproducible Design

- Calls: `100` from `results/call100_dataset/call100_manifest.csv`
- Presets: `6` (pentatonic_no_theory, pentatonic_balanced, pentatonic_conservative, pentatonic_creative, pentatonic_creative_wide, pentatonic_low_temp_no_strongbeat)
- Ablation variants: `7` (A0, A1, A2, A3, A4, A5, A6)
- Trials per preset/variant/call: `15`
- Expected samples: `63000`
- Actual rows: `63000`
- Validation status: `passed`

## Variant Definitions

- `A0 A0_raw_amt`: No external control.
- `A1 A1_prompt_cleaning`: Only tail-repeat compression / prompt cleaning is enabled.
- `A2 A2_repetition_suppression`: Adds generation-time repetition suppression / resampling.
- `A3 A3_duration_matching`: Adds response duration and note-count matching.
- `A4 A4_fallback`: Adds motif fallback for empty or invalid neural output.
- `A5 A5_style_constraint`: Adds pentatonic / two-bar / 4-4 response style constraints.
- `A6 A6_full_controlled`: All controlled-AMT modules are enabled.

## Summary By Variant

| variant_short | ablation_variant | sample_count | mean_objective_score | mean_cpr | mean_duration_match_ratio | mean_psr | mean_cadence_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | A0_raw_amt | 9000 | 0.560968 | 0.152200 | 0.823200 | 0.568857 | 0.427850 |
| A1 | A1_prompt_cleaning | 9000 | 0.574265 | 0.131736 | 0.822865 | 0.572559 | 0.440333 |
| A2 | A2_repetition_suppression | 9000 | 0.610704 | 0.036216 | 0.835172 | 0.538713 | 0.384656 |
| A3 | A3_duration_matching | 9000 | 0.640526 | 0.049203 | 0.826190 | 0.538235 | 0.384544 |
| A4 | A4_fallback | 9000 | 0.640526 | 0.049203 | 0.826190 | 0.538235 | 0.384544 |
| A5 | A5_style_constraint | 9000 | 0.708929 | 0.128617 | 0.844393 | 1.000000 | 0.731233 |
| A6 | A6_full_controlled | 9000 | 0.732610 | 0.135255 | 0.936885 | 1.000000 | 0.995389 |

## Module Contributions

| module_step | module_added | paired_sample_count | mean_delta | ci95_low | ci95_high | p_two_sided |
| --- | --- | --- | --- | --- | --- | --- |
| A1 minus A0 | prompt cleaning | 9000 | 0.013297 | 0.010832 | 0.015736 | 0.000000 |
| A2 minus A1 | repetition suppression | 9000 | 0.036439 | 0.034068 | 0.038919 | 0.000000 |
| A3 minus A2 | duration matching | 9000 | 0.029822 | 0.028143 | 0.031398 | 0.000000 |
| A4 minus A3 | motif fallback | 9000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| A5 minus A4 | style constraint | 9000 | 0.068403 | 0.066357 | 0.070537 | 0.000000 |
| A6 minus A5 | full controlled | 9000 | 0.023681 | 0.022480 | 0.024868 | 0.000000 |

## Validation

- No validation errors.
