# Optimization

This document describes the computational optimizations applied to the Unit 2 DP-SAR simulation study. The goal is to reduce runtime and improve numerical stability while preserving the statistical design and numerical results of the original simulation.

The original Unit 2 implementation is preserved as the baseline for regression testing and performance comparison.

---

## Optimization 1: Avoid Explicit Matrix Inversion in the DGP

### Problem identified

The SAR data-generating process generates the response according to

$$
Y=(I-\rho W)^{-1}(X\beta+E).
$$

In the Unit 2 implementation, the code explicitly formed the inverse of

$$
S=I-\rho W
$$

and then multiplied the inverse by the right-hand side.

The baseline implementation was:

```python
S = np.eye(N) - rho * W_dense
S_inv = np.linalg.inv(S)

rhs = X.dot(beta) + E
Y = S_inv.dot(rhs)
```

However, the full matrix inverse is not actually required. The quantity $Y$ is the solution of the linear system

$$
SY=X\beta+E.
$$

Therefore, explicitly constructing $S^{-1}$ performs unnecessary work and stores an additional dense $N\times N$ matrix.

### Solution implemented

The explicit inverse was replaced with a direct linear-system solve:

```python
S = np.eye(N) - rho * W_dense

rhs = X.dot(beta) + E

Y = np.linalg.solve(S, rhs)
```

The optimized implementation solves

$$
SY=X\beta+E
$$

directly without explicitly forming $S^{-1}$.

### Why this is preferable

Both dense matrix inversion and dense linear-system solution have approximately $O(N^3)$ asymptotic computational complexity, so this optimization does not change the big-O complexity of the DGP.

However, `np.linalg.solve` avoids explicitly constructing the entire inverse matrix. It therefore avoids unnecessary computation and memory allocation and is generally preferable numerically when the objective is only to solve a linear system.

The optimization is therefore primarily a constant-factor and numerical-stability improvement rather than an asymptotic-complexity improvement. The benchmark confirms that this constant-factor improvement is substantial in practice.

### Correctness validation

Regression tests compare the optimized solution

```python
np.linalg.solve(S, rhs)
```

with the original solution

```python
np.linalg.inv(S).dot(rhs)
```

using the same matrix and right-hand side.

The two results agree within floating-point tolerance.

An additional test verifies that the optimized result satisfies

$$
SY \approx X\beta+E.
$$

### Performance impact

To isolate the computational effect of replacing the explicit inverse, I benchmarked the two linear-algebra kernels directly:

```python
np.linalg.inv(S).dot(rhs)
```

versus

```python
np.linalg.solve(S, rhs)
```

for sample sizes $N=400,600,800,1000,1200$. Each configuration was repeated five times. Because short runtime measurements can be sensitive to operating-system scheduling and BLAS behavior, the median runtime is used as the primary summary.

| Sample size $N$ | Explicit inverse median (s) | Direct solve median (s) | Median speedup |
|---:|---:|---:|---:|
| 400 | 0.002892 | 0.001204 | 2.38× |
| 600 | 0.008078 | 0.003080 | 2.58× |
| 800 | 0.021661 | 0.008782 | 2.67× |
| 1000 | 0.031946 | 0.013710 | 2.44× |
| 1200 | 0.045448 | 0.014583 | 2.98× |

Across all tested sample sizes, the direct solve is consistently faster than explicitly constructing the inverse. The median speedup ranges from approximately 2.4× to 3.0×.

The improvement becomes especially relevant as $N$ increases because both methods operate on dense $N\times N$ matrices. Although both approaches retain approximately $O(N^3)$ asymptotic complexity, `np.linalg.solve` avoids explicitly constructing the full inverse matrix and therefore reduces the constant factor substantially.

This benchmark isolates only the linear-system computation. The end-to-end DGP speedup is smaller because the full data-generating process also includes network generation and simulation of $X$ and $E$.

### Trade-offs

The main trade-off is minimal. The optimized implementation still requires a dense $N\times N$ matrix and retains $O(N^3)$ computational complexity.

A sparse linear-system solver could potentially provide a larger performance improvement because the spatial weight matrix $W$ is sparse, but sparse solving was intentionally not introduced here. This keeps the optimization close to the Unit 2 implementation and isolates the benefit of replacing the explicit inverse with a direct dense solve.

Overall, this optimization provides a consistent 2.4×–3.0× speedup for the relevant linear-algebra kernel while also following standard numerical linear-algebra practice of solving the system directly rather than explicitly forming the matrix inverse.

---

## Optimization 2: Remove Redundant Matrix Operations in DP-QMLE

### Problem identified

Baseline profiling showed that the dominant computational cost was the proposed DP-QMLE estimator.

For a representative simulation with $N=800$, $\epsilon=1$, and one Monte Carlo replicate, the baseline profiling results were approximately:

| Component | Baseline runtime |
|---|---:|
| Full representative simulation | 1.551 s |
| DP-QMLE fit | 1.514 s |
| `f_lambda_ture_sig` | 1.483 s |
| `np.linalg.inv` | 0.836 s |
| NumPy dense `dot` operations | 0.537 s |

The DP-QMLE therefore accounted for almost all of the runtime of this representative simulation.

The score function `f_lambda_ture_sig` is evaluated repeatedly during the DP gradient iterations. For $N=800$, it was called 37 times.

The original implementation repeated several matrix operations during every score evaluation:

1. converting the same sparse weight matrix $W$ to a dense array;
2. recomputing matrix-vector products involving $W$ and $Y$;
3. explicitly computing the full dense matrix product

$$
W(I-\rho W)^{-1}
$$

only to obtain its trace.

These computations are unnecessary because several quantities remain constant across the gradient iterations.

---

### Optimization 2.1: Convert $W$ to dense form only once

The baseline score function repeatedly used

```python
W.toarray()
```

during every iteration.

Instead, the optimized estimator converts the matrix once before entering the iterative procedure:

```python
W_dense = (
    W.toarray()
    if sps.issparse(W)
    else np.asarray(W)
)
```

The same `W_dense` object is then passed to every score evaluation.

This eliminates repeated sparse-to-dense conversion.

---

### Optimization 2.2: Precompute $WY$

The baseline code computed the residual using

```python
hS = np.eye(N) - hrho * W_dense
V = hS.dot(Y) - X.dot(hbeta)
```

where

$$
hS=I-\rho W.
$$

Using the identity

$$
(I-\rho W)Y
=
Y-\rho WY,
$$

the same residual can be written as

```python
WY = W_dense.dot(Y)

V = (
    Y
    - hrho * WY
    - X.dot(hbeta)
)
```

Since $W$ and $Y$ remain fixed during the gradient iterations, $WY$ can be computed once and reused.

One subtlety is that the DP-QMLE algorithm truncates $Y$. The initial variance calculation uses the original response, while the gradient iterations use the truncated response. Therefore, $WY$ is recomputed once after truncation:

```python
WY_initial = W_dense.dot(Y)

# initial variance calculation
...

Y = np.clip(Y, -cy, cy)

# gradient iterations use truncated Y
WY = W_dense.dot(Y)
```

After that point, the same `WY` is reused for every iteration.

---

### Optimization 2.3: Avoid forming the full matrix product for the trace

The most important redundant dense matrix operation was

```python
np.trace(W_dense.dot(hS1))
```

where $hS1=(I-\rho W)^{-1}.$

The full $N\times N$ matrix product is not needed because only its trace is used.

Using the identity

$$
\operatorname{tr}(AB)=
\sum_{i,j} A_{ij}B_{ji},
$$

the trace can instead be computed as

```python
trace_term = np.sum(W_dense * hS1.T)
```

This avoids explicitly constructing the full matrix product.

The baseline trace calculation involves an $N\times N$ dense matrix multiplication with approximately $O(N^3)$ cost.
The optimized trace calculation uses an elementwise multiplication and summation with approximately $O(N^2)$ cost. The matrix inverse itself is still required, so the overall DP-QMLE asymptotic complexity remains dominated by the inverse.

---

### Before and after

#### Baseline

```python
hS = np.eye(N) - hrho * W.toarray()
hS1 = np.linalg.inv(hS)

V = (
    hS.dot(Y)
    - X.dot(hbeta).reshape(N, 1)
)

hsig = (V.T.dot(V) / N)[0, 0]

f_r = (
    -np.trace(
        W.toarray().dot(hS1)
    )
    + (1.0 / hsig)
    * V.T.dot(W.toarray()).dot(Y)
)
```

#### Optimized

```python
hS = np.eye(N) - hrho * W_dense
hS1 = np.linalg.inv(hS)

V = (
    Y
    - hrho * WY
    - X.dot(hbeta).reshape(N, 1)
)

hsig = (V.T.dot(V) / N)[0, 0]

trace_term = np.sum(
    W_dense * hS1.T
)

f_r = (
    -trace_term
    + (1.0 / hsig)
    * V.T.dot(WY)
)
```

---

### Performance impact

After Optimization 2, the same representative $N=800$, $\epsilon=1$ profiling workload produced:

| Component | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Full representative simulation | 1.551 s | 1.063 s | 31.5% |
| DP-QMLE fit | 1.514 s | 1.028 s | 32.1% |
| `f_lambda_ture_sig` | 1.483 s | 1.013 s | 31.7% |

The overall representative workload therefore achieved a speedup of approximately

$$
\frac{1.551}{1.063}
\approx 1.46\times.
$$

The DP-QMLE itself achieved approximately

$$
\frac{1.514}{1.028}
\approx 1.47\times
$$

speedup.

An important change in the profiling results is that dense matrix multiplication is no longer a major bottleneck.

In the baseline profile, NumPy dense `dot` operations consumed approximately 0.537 seconds. After the optimization, this term disappeared from the dominant profiling entries.

The remaining dominant operation is now matrix inversion:

```text
37 calls to np.linalg.inv
cumulative time: 0.886 seconds
```

Thus, after removing the redundant matrix operations, approximately

$$
\frac{0.886}{1.028}
\approx 86\%
$$

of the DP-QMLE runtime in this representative workload is associated with matrix inversion.

This makes the remaining computational bottleneck much clearer.

---

### Computational complexity

Before the optimization, each gradient evaluation contained two major cubic-cost operations:

1. the dense inverse of $I-\rho W$;
2. the dense product $W(I-\rho W)^{-1}$.

After the optimization, the second $O(N^3)$ operation is replaced by an $O(N^2)$ trace calculation. The inverse remains $O(N^3)$.

The number of DP-QMLE gradient iterations is approximately $T=0.8(log N)^2.$
Therefore, the dominant asymptotic complexity of the estimator remains approximately $O(N^3\log^2 N).$

The optimization does not change the overall big-O order, but it removes one expensive $O(N^3)$ matrix operation from every gradient evaluation and therefore substantially reduces the leading constant.

---

### Correctness validation

Two forms of regression testing are used.

First, the trace identity is tested directly:

```python
baseline = np.trace(
    W @ S_inv
)

optimized = np.sum(
    W * S_inv.T
)

np.testing.assert_allclose(
    baseline,
    optimized,
)
```

Second, the preserved Unit 2 DP-QMLE implementation is compared with the optimized implementation using the same generated data and identical random-number streams.

The resulting parameter estimates agree within floating-point tolerance.

This verifies that the optimization changes the computational implementation rather than the statistical estimator.

---

### Trade-offs

The optimized implementation introduces two precomputed quantities, `W_dense` and `WY`, that are passed into the score function.

This slightly increases code complexity because the score function has additional arguments and the distinction between the original and truncated versions of $Y$ must be handled carefully.

However, the additional code complexity is modest relative to the observed performance gain.

The optimization also retains a dense inverse at every DP-QMLE iteration. Therefore, the main asymptotic bottleneck remains and further algorithmic changes would be required to change the overall $O(N^3\log^2 N)$ scaling.

---

## Profiling evidence

The Unit 2 baseline profiling results are stored in:

```text
results/performance/baseline_profile.txt
```

The isolated DGP linear-solve benchmark is stored in:

`results/performance/dgp_solve_benchmark.csv`

The profiling results after Optimization 2 are stored in:

```text
results/performance/optimization2_profile.txt
```

The profiling comparison shows that the optimization substantially reduces time spent in repeated dense matrix operations and leaves `np.linalg.inv` as the dominant remaining computational bottleneck.

---

## Lessons learned

The profiling results were important because they showed that optimizing small Python-level operations would provide little benefit. Nearly all representative runtime was concentrated in the DP-QMLE score calculation and dense linear algebra.

The most useful optimization so far was not replacing NumPy matrix multiplication with different syntax, but instead recognizing that one large matrix multiplication did not need to be performed at all.

In particular,

$$
\operatorname{tr}(WS^{-1})
$$

can be computed without forming the complete matrix $WS^{-1}$. This produced a substantial runtime reduction while preserving the estimator.

The current profiling results also show that explicit matrix inversion is now the dominant remaining bottleneck. Rather than immediately introducing a more invasive change to the DP-QMLE mathematics, the next optimization stage focuses on simulation-level improvements such as eliminating repeated data generation and parallelizing independent Monte Carlo workloads.

---

## Optimization 3: Eliminate Repeated Matrix Inversion Using Eigenvalues

### Problem identified

After Optimization 2, repeated matrix inversion became the dominant remaining bottleneck in DP-QMLE.

For the representative $N=800$, $\epsilon=1$ workload before this optimization, DP-QMLE required approximately 1.028 seconds, and 37 calls to `np.linalg.inv` consumed approximately 0.886 seconds.

The inverse was required only for evaluating
$
\operatorname{tr}\left[
W(I-\rho W)^{-1}
\right].
$
Because the spatial weight matrix $W$ remains fixed throughout a DP-QMLE fit, repeatedly constructing and inverting
$
I-\rho W
$
at every gradient iteration is unnecessary.

---

### Solution implemented

Let $\lambda_1,\ldots,\lambda_N$ denote the eigenvalues of $W$. For $S=I-\rho W,$ the trace term can be written as $\text{tr}[
W(I-\rho W)^{-1}]
=\sum_{i=1}^{N}
\frac{\lambda_i}
{1-\rho\lambda_i}.$
Therefore, instead of explicitly computing $(I-\rho W)^{-1}$ at every iteration, the eigenvalues of $W$ are computed once at the beginning of the DP-QMLE fit:

```python id="vg5v6g"
W_eigenvalues = np.linalg.eigvals(
    W_dense
)
```

Each subsequent score evaluation computes the trace using only elementwise operations:

```python id="xt41bq"
trace_complex = np.sum(
    W_eigenvalues
    / (
        1.0
        - rho * W_eigenvalues
    )
)

trace_term = float(
    trace_complex.real
)
```

This removes the repeated dense matrix inversions from the gradient iterations.

---

### Before and after

#### Before Optimization 3

```python id="11kk8v"
hS = np.eye(N) - rho * W_dense

hS_inv = np.linalg.inv(hS)

trace_term = np.sum(
    W_dense * hS_inv.T
)
```

For $N=800$, the score function was evaluated 37 times, which resulted in 37 dense matrix inversions.

#### After Optimization 3

```python id="kvrjds"
W_eigenvalues = np.linalg.eigvals(
    W_dense
)
```

is evaluated once per DP-QMLE fit.

During each gradient iteration:

```python id="5wf54a"
trace_term = np.sum(
    W_eigenvalues
    / (
        1.0
        - rho * W_eigenvalues
    )
).real
```

is used instead.

Thus, many repeated dense inversions are replaced by one eigendecomposition followed by inexpensive vector operations.

---

### Computational complexity

Before this optimization, each score evaluation required an $N\times N$ dense matrix inverse with approximately
$
O(N^3)
$
computational cost.
The number of score evaluations grows approximately as $T=0.8(log N)^2.$
Therefore, the inverse-related component of one DP-QMLE fit had approximate complexity
$
O(N^3\log^2N).
$

After the optimization, the eigenvalues of $W$ are computed once at approximately
$
O(N^3)
$
cost.
Each subsequent trace calculation requires only operations on the $N$ eigenvalues, giving approximately
$
O(N)
$
work per iteration.
The trace-related computation therefore changes from approximately
$$
O(N^3\log^2N)
$$
to
$$
O(N^3 + N\log^2N).
$$

This is the first optimization in the project that changes the asymptotic structure of the dominant repeated DP-QMLE computation rather than only reducing its constant factor.

---

### Correctness validation

The eigenvalue-based trace was directly compared with the inverse-based trace calculation.

The previous implementation computes

```python id="9urjlo"
S = np.eye(N) - rho * W
S_inv = np.linalg.inv(S)

baseline = np.sum(
    W * S_inv.T
)
```

while the optimized implementation computes

```python id="tx89oa"
eigenvalues = np.linalg.eigvals(W)

optimized = np.sum(
    eigenvalues
    / (
        1.0
        - rho * eigenvalues
    )
)
```

The two calculations agree within floating-point tolerance.

The regression test was repeated across multiple values of $\rho$ within the feasible parameter region, including negative, zero, and positive values. This verifies that the identity is preserved across the range of spatial-autocorrelation parameters encountered during DP-QMLE optimization.

An estimator-level regression test additionally compares the pre-eigenvalue DP-QMLE with the optimized implementation under the same generated dataset and the same DP random-number stream.

---

### Numerical considerations

The Power-Law spatial weight matrix is not necessarily symmetric, so `np.linalg.eigvals` can return complex-valued eigenvalues.

Since $W$ is real, non-real eigenvalues occur in conjugate pairs. Consequently, the theoretical value

$$
\sum_i
\frac{\lambda_i}
{1-\rho\lambda_i}
$$

is real.

The implementation therefore checks the imaginary component of the calculated trace and uses its real part only when the remaining imaginary component is negligible numerical roundoff.

The DP-QMLE algorithm also constrains $|\rho|\le 0.5,$ which helps keep the denominators $1-\rho\lambda_i$ away from zero for the row-normalized spatial weight matrices considered in this study.

---

### Performance benchmark

The complete DP-QMLE estimator was benchmarked before and after the eigenvalue optimization for

$$
N\in\{400,600,800,1000,1200\}.
$$

For each sample size, both implementations used the same generated dataset and the same DP random-number stream. Each configuration was repeated five times, and median runtime is reported to reduce sensitivity to system scheduling and timing noise.

| Sample size $N$ | Pre-eigenvalue DP-QMLE median (s) | Eigenvalue DP-QMLE median (s) | Median speedup |
|---:|---:|---:|---:|
| 400  | 0.2636 | 0.0443 | 5.98× |
| 600  | 0.9061 | 0.1074 | 8.44× |
| 800  | 2.2333 | 0.2542 | 8.79× |
| 1000 | 4.4494 | 0.4070 | 10.93× |
| 1200 | 7.8362 | 0.5894 | 13.29× |

The eigenvalue formulation substantially improves DP-QMLE runtime at every tested sample size.

The reduction in median runtime is approximately:

| $N$ | Runtime reduction |
|---:|---:|
| 400 | 83.2% |
| 600 | 88.1% |
| 800 | 88.6% |
| 1000 | 90.9% |
| 1200 | 92.5% |

The speedup also increases with sample size, from approximately 6× at $N=400$ to more than 13× at $N=1200$.

This pattern is consistent with the theoretical motivation for the optimization. In the pre-eigenvalue implementation, an expensive $O(N^3)$ inverse was repeated at every gradient iteration. In the optimized implementation, only one $O(N^3)$ eigendecomposition is required, after which the repeated trace calculations are inexpensive.

---

### Profiling evidence

Profiling confirms that the computational structure of DP-QMLE changed as intended.

Before Optimization 3, the representative profile contained:

```text id="rmbxqc"
37 calls to np.linalg.inv
cumulative time: approximately 0.886 seconds
```

After Optimization 3, these repeated matrix inversions disappeared.

The optimized profile instead contains a single call to:

```text id="k3n3fz"
np.linalg.eigvals
```

while the 37 calls to `f_lambda_ture_sig` together require only a small amount of time outside of the eigendecomposition.

For example, in the profiling run:

```text id="7na2jd"
1 call   np.linalg.eigvals
37 calls f_lambda_ture_sig
```

were observed, confirming that the repeated inverse computation has been moved out of the iterative score evaluation.

The absolute runtime reported by `cProfile` is not used for the formal speedup calculation because profiling and benchmarking may use different BLAS threading and instrumentation conditions. The controlled five-repetition benchmark above is used for quantitative runtime comparisons.

---

### Performance interpretation

Optimization 3 provides a much larger speedup than the previous local algebraic optimizations.

At $N=800$, for example, median DP-QMLE runtime decreases from approximately $2.23\text{ seconds}$ to $0.25\text{ seconds},$ corresponding to approximately an $8.8\times$ speedup.

At $N=1200$, the median runtime decreases from approximately $7.84\text{ seconds}$ to $0.59\text{ seconds},$ corresponding to more than a $13\times$ speedup.

The increasing speedup with $N$ is consistent with the fact that the eliminated computation consisted of repeated dense $N\times N$ matrix inversions.

---

### Trade-offs

The major benefit is the elimination of the dominant repeated matrix inversions from DP-QMLE.

The main trade-off is additional numerical and conceptual complexity. The eigenvalue-based formula is less direct than explicitly forming $W(I-\rho W)^{-1}$, and non-symmetric spatial weight matrices can produce complex eigenvalues.

For this reason, additional regression tests and numerical safeguards are required.

The eigendecomposition itself remains an $O(N^3)$ operation, so the optimization does not eliminate cubic work completely. Instead, it replaces approximately $O$log^2 N)$ repeated cubic operations with one cubic operation per DP-QMLE fit.

The benchmark results indicate that this trade-off is highly favorable for the sample sizes considered in the simulation study.

## Optimization 4: Parallelize Independent Monte Carlo Replicates

### Problem identified

After the algorithmic optimizations, individual DP-QMLE fits became substantially faster. The remaining simulation study still contains many independent Monte Carlo workloads across sample sizes and replicates.

For a fixed DGP setting and replicate, the generated dataset is independent of other replicates. This makes the simulation naturally suitable for process-level parallelism.

The serial simulation therefore leaves available CPU resources unused when independent replicates are executed one after another.

---

### Solution implemented

The simulation was reorganized so that one parallel task corresponds to one DGP group and one Monte Carlo replicate.

Each worker performs the following operations:

1. generate one dataset for a fixed sample size and replicate;
2. reuse the same dataset for both privacy budgets;
3. fit DP-QMLE and the Wang--Song comparator for each privacy budget;
4. return the replicate-level results.

The parallel implementation uses Python's `ProcessPoolExecutor`.

A task therefore has the structure

```text
(N, replicate)
    |
    +-- generate dataset once
    |
    +-- epsilon = 1
    |      +-- DP-QMLE
    |      +-- Wang-Song
    |
    +-- epsilon = 2
           +-- DP-QMLE
           +-- Wang-Song
```

This task definition preserves the previous optimization that reuses the same generated dataset across privacy budgets.

---

### Why process-level parallelism was used

`ProcessPoolExecutor` was used instead of threads because the simulation performs CPU-intensive numerical linear algebra and also relies on NumPy and Python random-number generators.

Each worker process has an independent memory space and random-number-generator state, making process-level parallelism safer for the deterministic simulation design.

The existing `stable_seed` mechanism makes the random streams depend on the simulation identifiers rather than task execution order. Therefore, changing the scheduling order does not change the statistical results.

---

### Correctness validation

Serial and parallel executions were run using identical scenarios, replicate counts, and master random seeds.

The resulting output tables were sorted by

```text
N
epsilon
rep
method
parameter
```

and compared after excluding `elapsed_seconds`, since runtime measurements are expected to differ between serial and parallel execution.

The statistical outputs agree within numerical tolerance.

This verifies that parallel execution changes only the scheduling of the simulation and not the generated estimates or Monte Carlo design.

---

### Avoiding BLAS oversubscription

NumPy linear algebra routines may themselves use multiple CPU threads.

Running multiple Python processes while allowing every process to use multiple BLAS threads can create oversubscription, where the total number of active computational threads greatly exceeds the number of available CPU cores.

To isolate process-level parallelism, the benchmark was therefore run with

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

for every worker configuration.

---

### Performance benchmark

Parallel scaling was benchmarked using five Monte Carlo replicates for the full set of sample sizes and privacy budgets.

The same simulation workload was run using 1, 2, and 4 worker processes.

| Workers | Wall-clock time (s) | Speedup vs. 1 worker | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 17.32 | 1.00× | 100% |
| 2 | 13.54 | 1.28× | 64% |
| 4 | 14.79 | 1.17× | 29% |

Speedup is defined as

$$
S_p=\frac{T_1}{T_p},
$$

and parallel efficiency as

$$
E_p=\frac{S_p}{p}.
$$

For two workers,

$$
S_2
=
\frac{17.32}{13.54}
\approx1.28,
$$

corresponding to approximately 64% parallel efficiency.

For four workers,

$$
S_4
=
\frac{17.32}{14.79}
\approx1.17,
$$

corresponding to approximately 29% parallel efficiency.

---

### Interpretation

The two-worker configuration provided the best runtime among the tested configurations, reducing wall-clock time from 17.32 seconds to 13.54 seconds.

Increasing the worker count from two to four did not provide an additional improvement. Instead, runtime increased from 13.54 seconds to 14.79 seconds.

This indicates diminishing returns from process-level parallelism for the current workload.

Several factors can contribute to this behavior:

- process creation and task-dispatch overhead;
- inter-process serialization costs;
- competition for memory bandwidth and CPU cache;
- the relatively small number of tasks in the five-replicate benchmark;
- expensive linear-algebra operations such as eigendecomposition competing for shared hardware resources.

The increase in total CPU time also supports this interpretation. The measured user CPU time increased from approximately 16.41 seconds with one worker to 22.26 seconds with two workers and 32.15 seconds with four workers, while the wall-clock improvement did not scale proportionally.

Thus, adding more workers increases total computational work and contention even when it does not reduce elapsed time.

---

### Practical configuration

Based on the benchmark, two workers are the preferred parallel configuration for the current machine and simulation workload.

The Makefile therefore supports configurable worker counts, for example:

```bash
make parallel WORKERS=2
```

Using a configurable worker count is preferable to hard-coding the maximum number of available CPU cores because the best configuration depends on both the machine and the numerical workload.

---

### Trade-offs

Parallelization reduces wall-clock time without changing the statistical procedure, but it introduces additional implementation and runtime complexity.

The parallel runner requires process management, deterministic task-level random seeds, and safeguards against BLAS oversubscription.

Parallel speedup is also not linear in the number of workers. In the present benchmark, two workers improved runtime, while four workers were slower than two workers.

Therefore, parallelism should be treated as a tunable computational parameter rather than assuming that more workers always provide better performance.

For the current workload and hardware, two worker processes provide the best balance between parallel speedup and computational overhead.


## Final Performance Summary

### End-to-end runtime

The original Unit 2 simulation study required 788.29 seconds of wall-clock time for the complete serial pipeline with 50 Monte Carlo replicates per condition. After all algorithmic optimizations, the same complete serial pipeline required 342.37 seconds. Therefore, the final optimized implementation achieves an end-to-end speedup of $\frac{788.29}{342.37}\approx2.30\times.$ Equivalently, the total wall-clock runtime decreased by approximately $56.6\%.$ This comparison uses serial execution for both implementations, so the gain reflects the algorithmic and implementation optimizations rather than process-level parallelism.

### DP-QMLE runtime scaling

The isolated DP-QMLE benchmark shows a substantially larger improvement because DP-QMLE was the principal computational bottleneck in the Unit 2 implementation.

| Sample size $N$ | Unit 2 baseline median (s) | Final optimized median (s) | Median speedup |
|---:|---:|---:|---:|
| 400  | 0.2316 | 0.0260 | 8.90× |
| 600  | 0.8755 | 0.0654 | 13.47× |
| 800  | 2.1485 | 0.1469 | 14.63× |
| 1000 | 4.2919 | 0.2375 | 18.56× |
| 1200 | 7.5882 | 0.3521 | 21.52× |

The speedup increases with sample size. This is consistent with the main optimization: repeated dense matrix inversions were removed from the iterative DP-QMLE score calculation, so larger matrices benefit disproportionately.

### Empirical complexity

A log-log regression of median runtime against sample size gives empirical scaling exponents of approximately 3.17 for the Unit 2 baseline and 2.41 for the final optimized implementation.

Thus, over the range of sample sizes evaluated, $T_{\mathrm{baseline}}(N)\approx C_1 N^{3.17},$ whereas $T_{\mathrm{optimized}}(N)\approx C_2 N^{2.41}.$ The exponent 2.41 is an empirical scaling estimate rather than a theoretical complexity bound. Nevertheless, the substantial decrease is consistent with replacing repeated cubic-cost inversions by a single eigendecomposition followed by inexpensive iterative trace evaluations.

### Parallel scaling

Process-level parallelism was also evaluated after limiting each worker to one BLAS thread.

| Workers | Wall-clock time (s) | Speedup | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 17.32 | 1.00× | 100% |
| 2 | 13.54 | 1.28× | 64% |
| 4 | 14.79 | 1.17× | 29% |

Two workers provided the best performance on the test machine. Four workers increased CPU usage but did not further reduce wall-clock time, illustrating the effects of process overhead and hardware-resource contention.

### Numerical stability

The final 50-replicate simulation completed successfully for every combination of sample size, privacy budget, and method.

Across all 20 `(N, epsilon, method)` conditions, the number of failed fits was zero.

However, the Wang--Song Gaussian functional-mechanism estimator continued to exhibit occasional large but finite estimation errors. The largest observed absolute error was approximately 120.39 for the spatial parameter \$rho$ at $N=1200$ and \$epsilon=1$. Other large errors were also concentrated in the Wang--Song estimator.

These observations are consistent with the original statistical diagnostic plots: DP-QMLE remains tightly concentrated around the true parameters, whereas the Wang--Song estimator can generate extreme finite estimates.

Importantly, the computational optimizations did not remove or filter these extreme values. They remain part of the simulation output because numerical instability of an estimator is itself relevant finite-sample behavior.

### Overall conclusion

The largest performance gain came from changing the structure of the DP-QMLE trace calculation rather than from low-level syntax changes.

The progression of optimizations was:

1. avoid an unnecessary explicit inverse in the SAR data-generating process;
2. eliminate redundant matrix conversions and matrix products in DP-QMLE;
3. reuse identical generated datasets across matched privacy-budget scenarios;
4. replace repeated inverse-based trace calculations with a one-time eigendecomposition;
5. parallelize independent Monte Carlo workloads.

The final implementation reduces complete serial simulation runtime by approximately 56.6% while preserving the statistical design and numerical results of the original Unit 2 study.

The results also demonstrate that optimization should be guided by profiling. Early profiling identified repeated dense matrix inversion as the dominant bottleneck, and removing that bottleneck produced the largest performance improvement.