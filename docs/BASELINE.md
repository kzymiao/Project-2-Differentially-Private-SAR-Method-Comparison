# Baseline Performance

This document records the computational performance and numerical behavior of the original Unit 2 DP-SAR simulation study before the Unit 3 optimizations.

## Unit 2 Runtime

The full Unit 2 simulation uses 50 Monte Carlo replicates for each `(N, epsilon)` scenario, with $N\in\{400,600,800,1000,1200\}$ and $\epsilon\in\{1,2\}.$

The complete serial Unit 2 pipeline required:

| Timing measure | Time |
|---|---:|
| Wall-clock (`real`) | 788.29 s |
| CPU (`user`) | 5614.47 s |
| System (`sys`) | 202.59 s |

Thus, the original complete simulation required approximately $788.29/60 \approx 13.14$ minutes of wall-clock time.

The substantially larger CPU time relative to wall-clock time reflects the use of multithreaded numerical linear-algebra libraries underneath NumPy.

---

## Profiling Results

A representative baseline workload was profiled using

- $N=800$;
- $\epsilon=1$;
- one Monte Carlo replicate.

The dominant cumulative-time components were:

| Component | Cumulative time |
|---|---:|
| Full representative simulation | 1.551 s |
| DP-QMLE fit | 1.514 s |
| `f_lambda_ture_sig` | 1.483 s |
| `np.linalg.inv` | 0.836 s |
| NumPy dense `dot` operations | 0.537 s |

The DP-QMLE estimator therefore accounted for almost all of the runtime of this representative simulation.

Within DP-QMLE, nearly all computation was concentrated in the repeatedly evaluated score function `f_lambda_ture_sig`.

The baseline implementation repeatedly formed $S(\rho)=I-\rho W$ and computed the dense inverse $S(\rho)^{-1}.$

For $N=800$, the score function was evaluated approximately 37 times during one DP-QMLE fit. Dense matrix inversion and dense matrix multiplication were therefore the primary optimization targets.

The stored profiler output is available in:

```text
results/performance/baseline_profile.txt
```

---

## Computational Complexity

The dominant baseline DP-QMLE computation repeatedly performs dense $N\times N$ matrix inversion.

A dense inverse has approximately $O(N^3)$ computational complexity.

The number of DP-QMLE gradient iterations is approximately $T=0.8(\log N)^2.$

Therefore, the inverse-related portion of the baseline estimator has approximate theoretical complexity $O(N^3\log^2 N).$

The baseline implementation also explicitly formed dense matrix products involving $W(I-\rho W)^{-1},$

which introduce additional cubic-cost matrix operations.

### Empirical complexity

A log-log regression of median DP-QMLE runtime against sample size gives an empirical runtime exponent of approximately 3.17. Thus, over the tested sample sizes, $T_{\mathrm{baseline}}(N) \approx C N^{3.17}.$

The empirical behavior is therefore consistent with the approximately cubic dense linear-algebra operations identified by the theoretical and profiling analyses.

The corresponding benchmark and complexity results are stored in:

```text
results/performance/final_benchmark_summary.csv
results/performance/complexity_summary.csv
```

---

## Numerical Stability

The Unit 2 study did not discard unusually large finite estimates. Such values are retained because numerical instability is itself relevant finite-sample behavior.

The Wang--Song Gaussian functional-mechanism estimator exhibited substantially greater variability than DP-QMLE and occasionally produced very large but finite estimation errors. This behavior motivated the use of logarithmic scales in the RMSE figures and the inclusion of an estimation-error diagnostic plot.

In contrast, DP-QMLE estimates were substantially more concentrated around the true parameters in the evaluated settings.

The baseline study therefore exhibited numerical instability primarily through extreme finite Wang--Song estimates rather than through routine execution failures or non-finite outputs.

The Unit 3 stability analysis further quantifies these behaviors without filtering the extreme observations.

---

## Baseline Summary

The baseline analysis identifies three main computational issues:

1. repeated dense inversion of $I-\rho W$ during DP-QMLE iterations;
2. redundant dense matrix operations inside the score calculation;
3. repeated independent simulation work that can be reorganized or parallelized.

The original full serial pipeline required 788.29 seconds, and the empirical DP-QMLE runtime scaling exponent was approximately 3.17.

These observations motivate the optimizations documented in `docs/OPTIMIZATION.md`.