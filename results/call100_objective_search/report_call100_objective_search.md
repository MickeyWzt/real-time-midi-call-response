# Call100 Objective Search Report

## Experiment Scale

- Output directory: `results/call100_objective_search/` in this public release
- Calls: `100`
- Presets: `6`
- Candidates: `3`
- Trials per preset/candidate/call: `15`
- all_objective_results rows: `27000`
- Validation status: `passed`

## Dataset Distribution

### Source Dataset

- `POP909`: 50
- `calls_50`: 50

### Origin

- `artificial_stress`: 10
- `public_midi_extract`: 50
- `real_public_dataset`: 40

### Category

- `antecedent_half_cadence`: 1
- `arched_contour`: 1
- `blues_chromatic`: 1
- `chord_arpeggio_polyphonic`: 8
- `chromatic_outside`: 5
- `clear_motif_melody`: 12
- `dense_fast`: 5
- `dense_motion`: 1
- `expressive_rubato`: 8
- `false_ending_pause`: 4
- `large_leap_gap_fill`: 1
- `modal_question`: 1
- `pentatonic_clear_motif`: 1
- `real_two_bar_monophonic_major_minor`: 40
- `repetition_stress_test`: 1
- `repetition_tail`: 3
- `sparse_cadential`: 1
- `sparse_short`: 5
- `syncopation`: 1

## Top 10 Preset/Candidate

| rank | preset | candidate | sample_count | mean_objective_score | ci95_low_objective_score | ci95_high_objective_score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | pentatonic_balanced | motif_transform_baseline | 1500 | 0.7667 | 0.7628 | 0.7705 |
| 2 | pentatonic_conservative | motif_transform_baseline | 1500 | 0.7661 | 0.7621 | 0.7701 |
| 3 | pentatonic_conservative | amt_small_controlled | 1500 | 0.7643 | 0.7607 | 0.7680 |
| 4 | pentatonic_low_temp_no_strongbeat | amt_small_controlled | 1500 | 0.7628 | 0.7588 | 0.7668 |
| 5 | pentatonic_balanced | amt_small_controlled | 1500 | 0.7609 | 0.7569 | 0.7649 |
| 6 | pentatonic_creative | motif_transform_baseline | 1500 | 0.7582 | 0.7541 | 0.7622 |
| 7 | pentatonic_low_temp_no_strongbeat | motif_transform_baseline | 1500 | 0.7544 | 0.7504 | 0.7584 |
| 8 | pentatonic_creative_wide | motif_transform_baseline | 1500 | 0.7522 | 0.7482 | 0.7562 |
| 9 | pentatonic_creative | amt_small_controlled | 1500 | 0.7473 | 0.7435 | 0.7511 |
| 10 | pentatonic_no_theory | amt_small_controlled | 1500 | 0.7256 | 0.7210 | 0.7302 |

## Controlled vs Raw

Overall controlled-minus-raw mean difference: `0.063907` 95% CI `[0.061668, 0.066294]`, bootstrap p=`0.000000`.

## Motif Baseline vs Controlled

Overall controlled-minus-motif mean difference: `-0.005633` 95% CI `[-0.007882, -0.003501]`, bootstrap p=`0.000000`.

## Performance by Origin

| rank | origin | sample_count | mean_objective_score | ci95_low_objective_score | ci95_high_objective_score |
| --- | --- | --- | --- | --- | --- |
| 1 | artificial_stress | 2700 | 0.7428 | 0.7394 | 0.7462 |
| 2 | real_public_dataset | 10800 | 0.7394 | 0.7377 | 0.7412 |
| 3 | public_midi_extract | 13500 | 0.7152 | 0.7137 | 0.7166 |

## Performance by Category

| rank | category | sample_count | mean_objective_score | ci95_low_objective_score | ci95_high_objective_score |
| --- | --- | --- | --- | --- | --- |
| 1 | pentatonic_clear_motif | 270 | 0.7874 | 0.7774 | 0.7973 |
| 2 | modal_question | 270 | 0.7872 | 0.7789 | 0.7955 |
| 3 | syncopation | 270 | 0.7681 | 0.7563 | 0.7799 |
| 4 | blues_chromatic | 270 | 0.7600 | 0.7516 | 0.7685 |
| 5 | large_leap_gap_fill | 270 | 0.7485 | 0.7407 | 0.7562 |
| 6 | arched_contour | 270 | 0.7453 | 0.7348 | 0.7558 |
| 7 | dense_motion | 270 | 0.7430 | 0.7328 | 0.7533 |
| 8 | real_two_bar_monophonic_major_minor | 10800 | 0.7394 | 0.7377 | 0.7412 |
| 9 | repetition_tail | 810 | 0.7347 | 0.7284 | 0.7410 |
| 10 | clear_motif_melody | 3240 | 0.7341 | 0.7309 | 0.7374 |
| 11 | chord_arpeggio_polyphonic | 2160 | 0.7307 | 0.7284 | 0.7330 |
| 12 | chromatic_outside | 1350 | 0.7206 | 0.7165 | 0.7248 |
| 13 | dense_fast | 1350 | 0.7193 | 0.7152 | 0.7235 |
| 14 | repetition_stress_test | 270 | 0.7149 | 0.7048 | 0.7250 |
| 15 | antecedent_half_cadence | 270 | 0.7082 | 0.6979 | 0.7185 |
| 16 | false_ending_pause | 1080 | 0.7078 | 0.7015 | 0.7140 |
| 17 | expressive_rubato | 2160 | 0.6914 | 0.6879 | 0.6950 |
| 18 | sparse_short | 1350 | 0.6672 | 0.6626 | 0.6718 |
| 19 | sparse_cadential | 270 | 0.6653 | 0.6546 | 0.6759 |

## Performance by Sub Category

| rank | sub_category | sample_count | mean_objective_score | ci95_low_objective_score | ci95_high_objective_score |
| --- | --- | --- | --- | --- | --- |
| 1 | copied_from_call50 | 13500 | 0.7401 | 0.7385 | 0.7416 |
| 2 | two_bar_pop909_piano | 3240 | 0.7242 | 0.7220 | 0.7264 |
| 3 | two_bar_pop909_bridge | 5940 | 0.7134 | 0.7112 | 0.7156 |
| 4 | two_bar_pop909_melody | 4320 | 0.7108 | 0.7078 | 0.7138 |

## Failure Cases

- Failure-case rows written: `5377`
| reason | call_id | preset | candidate | trial | objective_score | duration_match_ratio | note_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bottom_5pct_objective_score | P039 | pentatonic_no_theory | amt_small_raw | 12 | 0.3701 | 0.8547 | 48.0000 |
| bottom_5pct_objective_score | P031 | pentatonic_creative_wide | amt_small_raw | 10 | 0.3714 | 0.9000 | 53.0000 |
| bottom_5pct_objective_score | P007 | pentatonic_no_theory | amt_small_raw | 10 | 0.3920 | 0.9375 | 32.0000 |
| bottom_5pct_objective_score | A04 | pentatonic_no_theory | amt_small_raw | 3 | 0.3959 | 0.9752 | 13.0000 |
| bottom_5pct_objective_score | P011 | pentatonic_balanced | amt_small_raw | 9 | 0.3960 | 0.8439 | 36.0000 |
| bottom_5pct_objective_score | P018 | pentatonic_creative_wide | amt_small_raw | 8 | 0.3981 | 0.8989 | 13.0000 |
| bottom_5pct_objective_score | P047 | pentatonic_no_theory | amt_small_raw | 12 | 0.3992 | 0.8325 | 36.0000 |
| bottom_5pct_objective_score | P039 | pentatonic_low_temp_no_strongbeat | amt_small_raw | 13 | 0.4008 | 0.9009 | 16.0000 |
| bottom_5pct_objective_score | P047 | pentatonic_no_theory | amt_small_raw | 5 | 0.4065 | 0.9100 | 47.0000 |
| bottom_5pct_objective_score | R39 | pentatonic_creative_wide | amt_small_raw | 12 | 0.4084 | 0.8602 | 43.0000 |
| bottom_5pct_objective_score | R39 | pentatonic_balanced | amt_small_raw | 7 | 0.4148 | 0.8351 | 30.0000 |
| bottom_5pct_objective_score | R39 | pentatonic_balanced | amt_small_raw | 12 | 0.4148 | 0.8351 | 30.0000 |
| bottom_5pct_objective_score | P039 | pentatonic_low_temp_no_strongbeat | amt_small_raw | 6 | 0.4150 | 0.9804 | 26.0000 |
| fallback_used;bottom_5pct_objective_score;poor_duration_match | P030 | pentatonic_no_theory | amt_small_controlled | 3 | 0.4159 | 0.4100 | 4.0000 |
| bottom_5pct_objective_score | R37 | pentatonic_balanced | amt_small_raw | 7 | 0.4160 | 0.9639 | 22.0000 |
| bottom_5pct_objective_score | P039 | pentatonic_no_theory | amt_small_raw | 7 | 0.4163 | 0.8421 | 15.0000 |
| bottom_5pct_objective_score | A07 | pentatonic_balanced | amt_small_raw | 4 | 0.4192 | 0.9250 | 13.0000 |
| bottom_5pct_objective_score | A07 | pentatonic_no_theory | amt_small_raw | 12 | 0.4198 | 0.9475 | 9.0000 |
| bottom_5pct_objective_score | R39 | pentatonic_low_temp_no_strongbeat | amt_small_raw | 7 | 0.4207 | 0.9324 | 28.0000 |
| bottom_5pct_objective_score | R39 | pentatonic_no_theory | amt_small_raw | 9 | 0.4215 | 0.8457 | 37.0000 |

## Statistical Confidence Intervals

- Summary CSV files use normal-approximation 95% CI over objective scores.
- `pairwise_bootstrap_tests.csv` uses paired bootstrap over preset/call/trial matched candidate differences.

## Validation Details

- No validation errors.
