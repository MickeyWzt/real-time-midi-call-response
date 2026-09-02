# Call100 MIDI-VAD Endpoint Benchmark

Reference boundary: final Note-Off in each isolated Call100 MIDI file.
Replay stops at the first commit. A first commit earlier than 100 ms before
the reference is a premature failure; commits after the first are never counted. Results are
reported at 0.5, 1.0, and 2.0 s post-boundary deadlines; the table below uses 2.0 s.
This file-end proxy is reproducible but is not a substitute for human boundary annotation.

PS-F1 is the custom premature-sensitive first-commit score: TP=S, FP=P,
and FN=P+L+M. Signed error is commit minus final Note-Off and is computed
only for successful commits. S/P/L/M are mutually exclusive and sum to 100.

| condition | precision | recall | PS-F1 [95% CI] | S/P/L/M | median error (s; N) | MAE (s) | cancel rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_full | 0.638 | 0.600 | 0.619 [0.523, 0.711] | 60/34/6/0 | 0.778 (60) | 0.853 | 0.083 |
| adaptive_no_clustering | 0.632 | 0.600 | 0.615 [0.520, 0.708] | 60/35/5/0 | 0.778 (60) | 0.837 | 0.099 |
| adaptive_no_confirmation | 0.571 | 0.560 | 0.566 [0.467, 0.663] | 56/42/2/0 | 0.623 (56) | 0.780 | 0.000 |
| adaptive_neither | 0.551 | 0.540 | 0.545 [0.444, 0.640] | 54/44/2/0 | 0.623 (54) | 0.756 | 0.000 |
| fixed_300ms | 0.130 | 0.130 | 0.130 [0.070, 0.200] | 13/87/0/0 | 0.418 (13) | 0.371 | 0.482 |
| fixed_500ms | 0.300 | 0.300 | 0.300 [0.210, 0.390] | 30/70/0/0 | 0.516 (30) | 0.462 | 0.200 |
| fixed_800ms | 0.500 | 0.500 | 0.500 [0.400, 0.590] | 50/50/0/0 | 0.778 (50) | 0.640 | 0.180 |
| cluster_40ms | 0.632 | 0.600 | 0.615 [0.520, 0.708] | 60/35/5/0 | 0.778 (60) | 0.844 | 0.083 |
| cluster_120ms | 0.638 | 0.600 | 0.619 [0.523, 0.711] | 60/34/6/0 | 0.778 (60) | 0.862 | 0.083 |
| confirm_75ms | 0.589 | 0.560 | 0.574 [0.472, 0.667] | 56/39/5/0 | 0.701 (56) | 0.799 | 0.029 |
| confirm_250ms | 0.688 | 0.640 | 0.663 [0.565, 0.751] | 64/29/7/0 | 0.895 (64) | 0.960 | 0.138 |
| theta_0p02 | 0.744 | 0.670 | 0.705 [0.613, 0.792] | 67/23/10/0 | 1.065 (67) | 1.156 | 0.091 |
| theta_0p10 | 0.530 | 0.530 | 0.530 [0.440, 0.630] | 53/47/0/0 | 0.605 (53) | 0.703 | 0.099 |
| theta_0p20 | 0.390 | 0.390 | 0.390 [0.300, 0.480] | 39/61/0/0 | 0.412 (39) | 0.466 | 0.310 |
| window_4 | 0.632 | 0.600 | 0.615 [0.521, 0.707] | 60/35/5/0 | 0.764 (60) | 0.842 | 0.099 |
| window_12 | 0.638 | 0.600 | 0.619 [0.523, 0.711] | 60/34/6/0 | 0.840 (60) | 0.890 | 0.074 |
| floor_0p125 | 0.638 | 0.600 | 0.619 [0.523, 0.711] | 60/34/6/0 | 0.778 (60) | 0.853 | 0.083 |
| floor_0p50 | 0.638 | 0.600 | 0.619 [0.523, 0.711] | 60/34/6/0 | 0.778 (60) | 0.853 | 0.083 |
| max_cutoff_2s | 0.650 | 0.650 | 0.650 [0.560, 0.740] | 65/35/0/0 | 0.805 (65) | 0.932 | 0.083 |
| max_cutoff_4s | 0.628 | 0.590 | 0.608 [0.510, 0.701] | 59/35/6/0 | 0.780 (59) | 0.862 | 0.083 |
