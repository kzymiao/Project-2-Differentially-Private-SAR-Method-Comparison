
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

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Testing and validation

Run the test suite with:

```bash
make PYTHON=python test
```

The current test suite contains four lightweight tests:

1. **Simulation design validation** (`test_final_design_crosses_sample_size_and_epsilon`): checks that the design contains the five intended sample sizes, both privacy budgets, and all ten scenarios.
2. **DGP verification** (`test_dgp_shapes_bounds_and_weight_rows`): checks the shapes of the generated data, verifies that covariates lie within the specified bounds, and checks row normalization of the network weight matrix for non-isolated nodes.
3. **Reproducibility check** (`test_dgp_reproducibility`): verifies that the same random seed produces identical


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

## Runtime

On a MacBook Pro, the full simulation with 50 Monte Carlo replicates
per condition takes approximately 14 minutes (13:48.25). Runtime will vary by
machine.

## Key Findings

Our simulation shows that the proposed DP-QMLE consistently achieves lower absolute bias and RMSE compared to the Wang–Song baseline, while exhibiting greater numerical stability across varying sample sizes and privacy budgets.