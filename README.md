
# DP-SAR Simulation Study
## Method background

The proposed DP-QMLE method is part of ongoing work on differentially private estimation and inference for spatial autoregressive models.

Proposed method:

Kong, Z., Tao, Y., Huang, D., and Cai, Z. Differentially Private Estimation and Inference for Spatial Autoregressive Models. Accept after minor revision on JMLR.

The comparison method is based on:

Wang, Z. and Song, Y. (2024). Privacy-preserving parametric inference for spatial autoregressive model. TEST, 33(3), 877–896.


This project compares the proposed differentially private QMLE for spatial autoregressive models with a Gaussian functional-mechanism baseline. The primary outcomes are parameter-wise **absolute bias** and RMSE as sample size and privacy budget change.



## Final simulation design

The final sample-size grid is

```text
N = 400, 600, 800, 1000, 1200
```

and the privacy budgets (The larger $\epsilon$ is, the weaker the privacy protection—intuitively, less noise is added.) are

```text
epsilon = 1, 2
```

with dimension of covariates `p=2`, true parameters of covariate `beta=(0.3,0.3)`, true network parameter `rho=0.2`, a Power-Law network, and another privacy budget parameter `delta=10/N^1.1`. The final run uses **50 Monte Carlo replicates per (N, epsilon) scenario**.

For a fixed `N` and replicate, `epsilon=1` and `epsilon=2` use the same generated dataset and paired method random streams.


## Setup

Create a virtual environment, install dependencies, run tests, and reproduce the complete simulation pipeline with:

```bash
make reproduce
```

This command creates a `.venv` virtual environment, installs the dependencies listed in `requirements.txt`, runs the test suite, and executes the complete simulation and analysis pipeline.

To run individual steps after setup:

```bash
make test
make all
```

## Reproducibility

The simulation uses a fixed default random seed (`1234`) and 50 Monte Carlo replicates per scenario. The random seed and number of replicates can be changed through the Makefile variables.

```bash
make PYTHON=.venv/bin/python all REPS=50 SEED=1234
```


## Quick pilot run

```bash
make PYTHON=python clean
make PYTHON=python simulate REPS=5
make PYTHON=python analyze
make PYTHON=python figures
```

## Final run

```bash
make PYTHON=python clean
make PYTHON=python simulate REPS=50
make PYTHON=python analyze
make PYTHON=python figures
```

or simply

```bash
make PYTHON=python clean
make PYTHON=python all
```

because the Makefile default is `REPS=50`.

## Outputs

- `results/raw/runs.csv`: replicate-level estimates and errors
- `results/summary.csv`: absolute bias, RMSE, and MCSE summaries by `N`, `epsilon`, method, and parameter
- `results/figures/absolute_bias_vs_N.png`: absolute bias versus sample size, with separate rows for epsilon=1 and epsilon=2
- `results/figures/rmse_vs_N.png`: RMSE versus sample size, with separate rows for epsilon=1 and epsilon=2
- `results/figures/epsilon_rmse_comparison.png`: direct RMSE comparison of epsilon=1 versus epsilon=2 for each method
- `results/figures/error_distribution_diagnostic.png`: estimation-error distributions at the largest N under the stronger privacy setting epsilon=1

## Interpretation note

Bias is defined as `abs(mean(theta_hat - theta))`, so all reported bias values are nonnegative. Error bars around bias are approximate Monte Carlo uncertainty bars based on the MCSE of the underlying mean estimation error, with the lower plotted whisker clipped at zero.

Occasional large finite estimates are retained rather than filtered, because numerical instability is part of finite-sample utility. The RMSE plots therefore use a log scale so both small and large errors remain visible.

## Key Findings

Our simulation shows that the proposed DP-QMLE consistently achieves lower absolute bias and RMSE compared to the Wang–Song baseline, while exhibiting greater numerical stability across varying sample sizes and privacy budgets.

## Unit 3: High-Performance Simulation Study

The Unit 3 project optimizes the original Unit 2 simulation study for computational efficiency and numerical stability while preserving the statistical design.

The main optimizations are:

1. replacing an explicit inverse in SAR data generation with a direct linear solve;
2. eliminating redundant matrix conversions and matrix products in DP-QMLE;
3. reusing generated datasets across matched privacy-budget scenarios;
4. replacing repeated inverse-based DP-QMLE trace calculations with a one-time eigendecomposition;
5. parallelizing independent Monte Carlo workloads.

Detailed performance analysis is available in:

- `docs/BASELINE.md`
- `docs/OPTIMIZATION.md`

### Performance commands

```bash
make profile
make benchmark
make complexity
make parallel WORKERS=2
make stability-check
```

The two-worker configuration is the default recommended parallel setting on the machine used for this project.

## Performance results

For the complete 50-replicate serial simulation pipeline:

| Implementation | Wall-clock time |
|---|---:|
| Unit 2 baseline | 788.29 s |
| Final optimized | 342.37 s |

The final serial implementation therefore achieves approximately a **2.30× end-to-end speedup**, corresponding to a **56.6% reduction in wall-clock runtime**.

The isolated DP-QMLE benchmark shows larger gains, with median speedup increasing from approximately **8.9× at \(N=400\)** to **21.5× at \(N=1200\)**.

Empirical log-log runtime scaling decreased from an exponent of approximately **3.17** for the Unit 2 implementation to approximately **2.41** for the optimized implementation.

## Testing and validation

Run all automated tests with:

```bash
make PYTHON=python test
```

The test suite includes:

- simulation-design validation;
- DGP shape, boundedness, and normalization checks;
- random-seed reproducibility;
- metric validation;
- regression tests comparing direct linear solving with the original inverse-based calculation;
- trace-identity validation;
- eigenvalue-based trace validation across multiple values of \(\rho\);
- estimator-level regression tests comparing optimized and preserved baseline implementations;
- verification that datasets are reused across privacy budgets;
- serial-versus-parallel result equivalence.

## Numerical stability

The final 50-replicate study completed without failed fits in any `(N, epsilon, method)` condition.

The Wang--Song Gaussian functional-mechanism baseline nevertheless produced several large finite estimation errors. These values are retained rather than filtered because numerical instability is part of the finite-sample comparison.

DP-QMLE remained substantially more stable in the final error-distribution diagnostics.