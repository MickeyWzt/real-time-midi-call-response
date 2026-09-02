# Call--Response Feature Association Analysis

This analysis measures reproducible symbolic feature association between each Call100 input and its response. It is not a measure of musical quality, listener preference, harmonic appropriateness, perceptual turn taking, or live co-performance usefulness.

## Pairing and null construction

All 27,000 harmonized rows were resolved through the existing local answer keys. Every result row stores only MIDI filenames and SHA-256 values; local filesystem paths are not published. For every `(preset, candidate, trial)` block, the observed pair is compared with the mean of all valid mismatched response pairings in the same 100-Call block. The per-row `*_mismatched_valid_count` records the dynamic denominator; an unavailable matched score or zero valid mismatches yields `NA`. Association excess is `actual feature similarity - mean valid-mismatched feature similarity`.

For each candidate and metric, association statistics are computed separately in all six-preset-by-15-trial blocks and then averaged within each Call. The primary 95% interval is a classical leave-one-Call-out jackknife normal interval. Every deletion removes the matched Call--response unit and rebuilds every retained Call's mismatch bank in every block before recomputing and aggregating association excess. The 87-source-cluster point and interval use the analogous leave-one-source-cluster-out jackknife with the same equal-source estimand as the other 87-source analyses: average valid Calls within each retained source, then average retained source means with equal weight. Candidate contrasts use the same paired Call/source deletions, rebuild both banks in every replicate, and retain a Call only when both candidate excesses are defined after within-Call aggregation. No hypothesis-test p-values are reported.

## Metrics

- `contour_r`: Pearson correlation of 16-point linearly event-index-resampled onset-cluster median-pitch contours; vectors are centered and L2-normalized independently. At least three clusters and centered norm above `1e-12` are required.
- `interval_2gram_dice`: multiset Dice `2*sum(min(count_call, count_response))/(ngrams_call+ngrams_response)` over adjacent signed interval-pairs. Median-pitch differences use Python ties-to-even integer rounding before clipping to [-12, 12] semitones; at least three clusters are required.
- `onset_profile_cosine`: cosine similarity of 16-bin onset-cluster density profiles after normalizing each phrase by its own onset-span. Bin index is `min(15, floor(16*(t-t0)/(tlast-t0)))`; counts are L2-normalized, and at least two clusters with span above `1e-12` are required. Duration matching and grid projection can affect this feature, so it remains a control/association check only.

Short or degenerate phrases receive `NA` for a metric rather than a synthetic zero; coverage is reported below.

## Candidate association-excess summaries

| metric | candidate | valid_trial_rows | trial_coverage | call_count | association_excess_mean | association_excess_mean_call_jackknife_ci95_low | association_excess_mean_call_jackknife_ci95_high | association_excess_source_mean | association_excess_mean_source_jackknife_ci95_low | association_excess_mean_source_jackknife_ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|
| contour_r | amt_small_raw | 8807 | 0.978556 | 100 | -0.014669 | -0.047801 | 0.018463 | -0.015747 | -0.050464 | 0.018969 |
| contour_r | amt_small_controlled | 8980 | 0.997778 | 100 | -0.043639 | -0.093617 | 0.006339 | -0.047923 | -0.097717 | 0.001871 |
| contour_r | motif_transform_baseline | 8939 | 0.993222 | 100 | -0.647758 | -0.690848 | -0.604667 | -0.644255 | -0.686915 | -0.601595 |
| interval_2gram_dice | amt_small_raw | 9000 | 1.000000 | 100 | 0.138600 | 0.117683 | 0.159517 | 0.151910 | 0.129195 | 0.174626 |
| interval_2gram_dice | amt_small_controlled | 8986 | 0.998444 | 100 | 0.041885 | 0.029691 | 0.054079 | 0.047567 | 0.033219 | 0.061914 |
| interval_2gram_dice | motif_transform_baseline | 8940 | 0.993333 | 100 | 0.014844 | 0.004651 | 0.025037 | 0.016952 | 0.004682 | 0.029221 |
| onset_profile_cosine | amt_small_raw | 9000 | 1.000000 | 100 | 0.009886 | 0.004497 | 0.015275 | 0.011671 | 0.005100 | 0.018242 |
| onset_profile_cosine | amt_small_controlled | 9000 | 1.000000 | 100 | 0.029901 | 0.017318 | 0.042484 | 0.031829 | 0.016606 | 0.047052 |
| onset_profile_cosine | motif_transform_baseline | 9000 | 1.000000 | 100 | 0.152945 | 0.127009 | 0.178882 | 0.156143 | 0.126072 | 0.186214 |

## Call- and source-level candidate contrasts on association excess

| metric | comparison | call_count | mean_difference | mean_difference_call_jackknife_ci95_low | mean_difference_call_jackknife_ci95_high | source_cluster_mean_difference | mean_difference_source_jackknife_ci95_low | mean_difference_source_jackknife_ci95_high |
|---|---|---|---|---|---|---|---|---|
| contour_r | controlled_minus_raw | 100 | -0.028970 | -0.064251 | 0.006311 | -0.032176 | -0.066140 | 0.001789 |
| contour_r | controlled_minus_motif | 100 | 0.604119 | 0.546297 | 0.661941 | 0.596332 | 0.537356 | 0.655309 |
| contour_r | motif_minus_raw | 100 | -0.633089 | -0.681071 | -0.585107 | -0.628508 | -0.678485 | -0.578531 |
| interval_2gram_dice | controlled_minus_raw | 100 | -0.096715 | -0.113257 | -0.080172 | -0.104344 | -0.122009 | -0.086678 |
| interval_2gram_dice | controlled_minus_motif | 100 | 0.027041 | 0.015330 | 0.038753 | 0.030615 | 0.017742 | 0.043488 |
| interval_2gram_dice | motif_minus_raw | 100 | -0.123756 | -0.145381 | -0.102131 | -0.134958 | -0.158577 | -0.111340 |
| onset_profile_cosine | controlled_minus_raw | 100 | 0.020016 | 0.006927 | 0.033105 | 0.020158 | 0.006123 | 0.034193 |
| onset_profile_cosine | controlled_minus_motif | 100 | -0.123044 | -0.149430 | -0.096658 | -0.124314 | -0.153121 | -0.095507 |
| onset_profile_cosine | motif_minus_raw | 100 | 0.143060 | 0.116104 | 0.170016 | 0.144472 | 0.114379 | 0.174565 |

## Same-source mismatch exclusion sensitivity

The primary mismatch bank includes all other Calls in the same condition, including another excerpt from the same source song. As a sensitivity analysis, the table below removes the 40 ordered off-diagonal cells per 100-Call block whose response source cluster matches the focal Call, then rebuilds every mismatch bank and repeats the paired Call/source jackknives. The matched diagonal is retained. This exclusion does not change the direction or interval interpretation of any controlled-minus-raw contrast.

| metric | comparison | call_count | mean_difference | mean_difference_call_jackknife_ci95_low | mean_difference_call_jackknife_ci95_high | source_cluster_mean_difference | mean_difference_source_jackknife_ci95_low | mean_difference_source_jackknife_ci95_high |
|---|---|---|---|---|---|---|---|---|
| contour_r | controlled_minus_raw | 100 | -0.028909 | -0.064386 | 0.006568 | -0.032178 | -0.066161 | 0.001805 |
| interval_2gram_dice | controlled_minus_raw | 100 | -0.096742 | -0.113273 | -0.080210 | -0.104352 | -0.122015 | -0.086690 |
| onset_profile_cosine | controlled_minus_raw | 100 | 0.020266 | 0.006888 | 0.033643 | 0.020212 | 0.006102 | 0.034323 |

## Interpretation boundary

Positive association excess means that the selected symbolic feature is more similar for the observed Call--response pairing than for same-condition mismatches in this finite development benchmark. It does not establish musical quality, conversational appropriateness, generalization, human-perceived contingency, or causal benefit of a module.
