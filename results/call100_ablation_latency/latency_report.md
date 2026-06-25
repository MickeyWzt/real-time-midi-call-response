# Call100 Realtime Latency Logging Study

## Design

- L0 preload off: inference starts at endpoint commit.
- L1 preload on: inference is modeled as starting during the candidate-endpoint confirmation window.
- The logged inference timings come from actual local AMT generation during the A6 full-controlled ablation runs.
- MIDI output timing is represented by the local playback scheduler model with the configured micro-buffer.

## Summary By Condition

| condition | sample_count | mean_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | max_latency_ms | underrun_rate | mean_first_token_latency_ms | mean_total_generation_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_preload_off | 9000 | 161.302711 | 136.455450 | 323.520665 | 479.574320 | 815.510000 | 0.207444 | 81.302711 | 1655.312037 |
| L1_preload_on | 9000 | 85.534258 | 80.000000 | 80.000000 | 229.574320 | 565.510000 | 0.035556 | 81.302711 | 1655.312037 |

## Preload Comparison

| comparison | paired_sample_count | mean_latency_reduction_ms | ci95_low | ci95_high | p_two_sided |
| --- | --- | --- | --- | --- | --- |
| L1_preload_on_vs_L0_preload_off | 9000 | 75.768453 | 74.723760 | 76.792700 | 0.000000 |
