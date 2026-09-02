# Call100 First-Event Readiness Replay

The source timing is elapsed local AMT inference until the first complete `GeneratedEvent` is yielded; its legacy field name is `first_token_latency_sec`. The commit-anchored quantity `max(B, C1-H)` is an earliest readiness/release lower bound, not measured MIDI-send, host-receive, or audio-onset latency. Full-stream underrun is also unmeasured.

## Summary By Condition

| condition | sample_count | mean_first_event_readiness_lower_bound_ms | p50_first_event_readiness_lower_bound_ms | p95_first_event_readiness_lower_bound_ms | startup_deadline_miss_rate | mean_first_generated_event_latency_ms | mean_total_generation_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_preload_off | 9000 | 101.543478 | 80.000000 | 243.520665 | 0.207444 | 81.302711 | 1655.312037 |
| L1_preload_on | 9000 | 86.544816 | 80.000000 | 93.520665 | 0.052444 | 81.302711 | 1655.312037 |

## Paired Preload Comparison

| comparison | paired_sample_count | mean_readiness_lower_bound_reduction_ms | positive_pairs | negative_pairs | tied_pairs | inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1_preload_on_vs_L0_preload_off | 9000 | 14.998662 | 1867 | 0 | 7133 | deterministic replay; no hypothesis test |
