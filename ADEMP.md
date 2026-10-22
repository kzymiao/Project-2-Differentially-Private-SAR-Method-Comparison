# ADEMP: Differentially Private SAR Method Comparison


## Aims
Evaluate how estimation accuracy changes with sample size and privacy budget, and compare the proposed DP-QMLE with a functional-mechanism baseline under matched approximate privacy parameters. The primary questions are whether DP-QMLE achieves lower finite-sample absolute bias and RMSE, how the performance gap changes with sample size, and how utility changes when the privacy budget increases from epsilon=1 to epsilon=2. The larger $\epsilon$ is, the weaker the privacy protection—intuitively, less noise is added.

## Data-generating mechanisms
For each replicate, generate SAR data using

$$Y=(I-\rho W)^{-1}(X\beta+E).$$

This is a spatial autoregressive (SAR) model. $Y$ is the $N \times 1$ vector of outcomes, $X$ is the $N \times p$ covariate matrix, $\beta$ is the $p \times 1$ coefficient vector, and $E$ is the $N \times 1$ vector of structural errors. $W$ is the $N \times N$ spatial weight matrix encoding network/spatial connections, and $\rho$ is the spatial autocorrelation parameter. 

We use the existing `generate_compare` design: Power-Law network, `p=2`, `beta=(0.3,0.3)`, `rho=0.2`, covariates from `Uniform(-1,1)`, and Gaussian structural errors with variance `0.5`.

The simulation varies

- `N` over 400, 600, 800, 1000, and 1200;
- `epsilon` over 1 and 2.

For every scenario, use `delta=10/N^1.1`. Both methods are fit to the same dataset within each replicate. For a fixed N and replicate, the same data and paired random streams are used across epsilon values to reduce Monte Carlo noise in the privacy-budget comparison.

## Estimands / targets
The target parameter vector is `(beta1, beta2, rho)`.

## Methods

1. **DP-QMLE (proposed method).**
   An iterative differentially private quasi-maximum likelihood
   estimator for the SAR model.

2. **Wang–Song Gaussian Functional Mechanism baseline.**
   A Gaussian-mechanism variant of the functional-mechanism estimator
   used as an approximate-DP comparator.

The two methods are evaluated on the same generated dataset within
each Monte Carlo replicate and under the same nominal epsilon and
delta settings.

## Performance measures
For each `(N, epsilon, method, parameter)` combination, report:

- absolute empirical bias, `abs(mean(theta_hat - theta))`;
- RMSE, `sqrt(mean((theta_hat - theta)^2))`;
- Monte Carlo uncertainty for both summaries;
- successful-fit counts and raw replicate-level estimates.

Bias is nonnegative in both the summary table and figures. The plotted lower uncertainty whisker is clipped at zero. RMSE is shown on a log scale when needed to display both methods clearly in the presence of occasional large errors.

### Design matrix

| N | p | beta | rho | network | epsilon | delta | final replicates |
|---:|---:|---|---:|---|---:|---|---:|
| 400  | 2 | (0.3,0.3) | 0.2 | PowerLaw | 1.0 | 10/N^1.1 | 50 |
| 600  | 2 | (0.3,0.3) | 0.2 | PowerLaw | 1.0 | 10/N^1.1 | 50 |
| 800  | 2 | (0.3,0.3) | 0.2 | PowerLaw | 1.0 | 10/N^1.1 | 50 |
| 1000 | 2 | (0.3,0.3) | 0.2 | PowerLaw | 1.0 | 10/N^1.1 | 50 |
| 1200 | 2 | (0.3,0.3) | 0.2 | PowerLaw | 1.0 | 10/N^1.1 | 50 |
| 400  | 2 | (0.3,0.3) | 0.2 | PowerLaw | 2.0 | 10/N^1.1 | 50 |
| 600  | 2 | (0.3,0.3) | 0.2 | PowerLaw | 2.0 | 10/N^1.1 | 50 |
| 800  | 2 | (0.3,0.3) | 0.2 | PowerLaw | 2.0 | 10/N^1.1 | 50 |
| 1000 | 2 | (0.3,0.3) | 0.2 | PowerLaw | 2.0 | 10/N^1.1 | 50 |
| 1200 | 2 | (0.3,0.3) | 0.2 | PowerLaw | 2.0 | 10/N^1.1 | 50 |
