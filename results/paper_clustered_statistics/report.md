# Cluster-aware paper statistics

Trial rows are descriptive. Primary uncertainty is computed after averaging within each of the 100 calls. Source-cluster sensitivity uses 87 independent clusters (77 POP909 songs and 10 artificial stress calls).

## Candidate means

| candidate | mean_objective_score | call_cluster_ci95_low | call_cluster_ci95_high |
| --- | --- | --- | --- |
| amt_small_controlled | 0.732610 | 0.722531 | 0.742323 |
| amt_small_raw | 0.560968 | 0.547154 | 0.575198 |
| motif_transform_baseline | 0.752682 | 0.741863 | 0.763309 |

## Candidate comparisons

| comparison | mean_difference | call_cluster_ci95_low | call_cluster_ci95_high | positive_call_fraction | source_cluster_ci95_low | source_cluster_ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| controlled_minus_raw | 0.171642 | 0.157901 | 0.185151 | 1.000000 | 0.158728 | 0.187555 |
| controlled_minus_motif | -0.020072 | -0.031425 | -0.008749 | 0.400000 | -0.032944 | -0.008721 |
| motif_minus_raw | 0.191714 | 0.174714 | 0.208189 | 0.960000 | 0.176257 | 0.211206 |

## Stepwise ablation (SCC only)

| comparison | mean_difference | call_cluster_ci95_low | call_cluster_ci95_high | positive_call_fraction |
| --- | --- | --- | --- | --- |
| A1_minus_A0 | 0.013297 | 0.007294 | 0.019632 | 0.540000 |
| A2_minus_A1 | 0.036439 | 0.028276 | 0.045060 | 0.850000 |
| A3_minus_A2 | 0.029822 | 0.022882 | 0.036700 | 0.920000 |
| A4_minus_A3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| A5_minus_A4 | 0.068403 | 0.056710 | 0.080175 | 0.910000 |
| A6_minus_A5 | 0.023681 | 0.018953 | 0.028472 | 0.910000 |

## Scheduler replay

| condition | mean_first_event_readiness_lower_bound_ms | call_cluster_ci95_low_ms | call_cluster_ci95_high_ms | startup_deadline_miss_rate |
| --- | --- | --- | --- | --- |
| L0_preload_off | 101.543478 | 93.111207 | 111.435739 | 0.207444 |
| L1_preload_on | 86.544816 | 82.797564 | 91.019965 | 0.052444 |

| comparison | mean_difference | call_cluster_ci95_low | call_cluster_ci95_high | positive_call_fraction |
| --- | --- | --- | --- | --- |
| preload_readiness_lower_bound_reduction_ms | 14.998662 | 10.151631 | 20.493798 | 0.980000 |
| startup_deadline_miss_rate_reduction | 0.155000 | 0.124889 | 0.189000 | 0.970000 |

The 80 ms target is anchored to endpoint commit. Full streaming underrun requires per-event readiness timestamps and is not estimated by these data.
