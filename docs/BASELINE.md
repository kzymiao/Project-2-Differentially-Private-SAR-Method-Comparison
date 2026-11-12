# Baseline Performance

## Unit 2 runtime

The full Unit 2 simulation uses 50 Monte Carlo replicates
for each (N, epsilon) scenario.

Total runtime:

- real time: XXX seconds

## Profiling results

Representative condition:

- N = 800
- epsilon = 1
- one Monte Carlo replicate

Main cumulative-time bottlenecks:

| Rank | Function | Observation |
| --- | --- | --- |
| 1 | ... | ... |
| 2 | ... | ... |
| 3 | ... | ... |

## Complexity analysis

The dominant DP-QMLE computation repeatedly forms and
inverts an N x N matrix.

A dense inverse has approximately O(N^3) complexity.
Since the algorithm performs approximately
T = 0.8(log N)^2 iterations, the dominant theoretical
cost is approximately O(N^3 log^2 N).

## Numerical stability

Record failures, warnings, and unusually large finite
estimates observed in the baseline runs.
