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
np.trace(
    W_dense.dot(hS1)
)
```

where

$$
hS1=(I-\rho W)^{-1}.
$$

The full $N\times N$ matrix product is not needed because only its trace is used.

Using the identity

$$
\operatorname{tr}(AB)
=
\sum_{i,j} A_{ij}B_{ji},
$$

the trace can instead be computed as

```python
trace_term = np.sum(
    W_dense * hS1.T
)
```

This avoids explicitly constructing the full matrix product.

The baseline trace calculation involves an $N\times N$ dense matrix multiplication with approximately

$$
O(N^3)
$$

cost.

The optimized trace calculation uses an elementwise multiplication and summation with approximately

$$
O(N^2)
$$

cost.

The matrix inverse itself is still required, so the overall DP-QMLE asymptotic complexity remains dominated by the inverse.

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

After the optimization, the second $O(N^3)$ operation is replaced by an $O(N^2)$ trace calculation.

The inverse remains $O(N^3)$.

The number of DP-QMLE gradient iterations is approximately

$$
T=0.8(\log N)^2.
$$

Therefore, the dominant asymptotic complexity of the estimator remains approximately

$$
O(N^3\log^2 N).
$$

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

$$
\operatorname{tr}\left[
W(I-\rho W)^{-1}
\right].
$$

Because the spatial weight matrix $W$ remains fixed throughout a DP-QMLE fit, repeatedly constructing and inverting

$$
I-\rho W
$$

at every gradient iteration is unnecessary.

---

### Solution implemented

Let

$$
\lambda_1,\ldots,\lambda_N
$$

denote the eigenvalues of $W$.

For

$$
S(\rho)=I-\rho W,
$$

the trace term can be written as

$$
\operatorname{tr}\left[
W(I-\rho W)^{-1}
\right]
=
\sum_{i=1}^{N}
\frac{\lambda_i}
{1-\rho\lambda_i}.
$$

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

$$
O(N^3)
$$

computational cost.

The number of score evaluations grows approximately as

$$
T=0.8(\log N)^2.
$$

Therefore, the inverse-related component of one DP-QMLE fit had approximate complexity

$$
O(N^3\log^2N).
$$

After the optimization, the eigenvalues of $W$ are computed once at approximately

$$
O(N^3)
$$

cost.

Each subsequent trace calculation requires only operations on the $N$ eigenvalues, giving approximately

$$
O(N)
$$

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

The DP-QMLE algorithm also constrains

$$
|\rho|\le 0.5,
$$

which helps keep the denominators

$$
1-\rho\lambda_i
$$

away from zero for the row-normalized spatial weight matrices considered in this study.

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

The eigendecomposition itself remains an $O(N^3)$ operation, so the optimization does not eliminate cubic work completely. Instead, it replaces approximately $O(\log^2 N)$ repeated cubic operations with one cubic operation per DP-QMLE fit.

The benchmark results indicate that this trade-off is highly favorable for the sample sizes considered in the simulation study.