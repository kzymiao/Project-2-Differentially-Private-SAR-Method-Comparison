# ANALYSIS

## Main findings

Across the simulation settings, the proposed **DP-QMLE** generally has smaller absolute bias and RMSE than the Wang–Song Gaussian functional-mechanism baseline for all three parameters, $\beta_1$, $\beta_2$, and $\rho$. The difference is especially clear in the RMSE plots: DP-QMLE remains comparatively stable as the sample size increases, whereas the Wang–Song estimator is much more variable and occasionally produces very large errors. The diagnostic boxplots at $N=1200$ and $\epsilon=1$ reinforce this point: the DP-QMLE errors are tightly concentrated near zero, while the Wang–Song method exhibits substantially wider distributions and several extreme outliers.

The privacy-budget comparison shows the expected privacy–utility tradeoff. Increasing $\epsilon$ from 1 to 2 weakens privacy protection and reduces the amount of perturbation, and the estimation accuracy generally improves for both methods. In particular, RMSE is usually smaller at $\epsilon=2$ than at $\epsilon=1$, with the improvement being especially visible for the Wang–Song baseline. DP-QMLE nevertheless remains the more accurate and stable method across the sample sizes considered.

## Interpretation

The sample-size trends are also broadly consistent with statistical intuition. For DP-QMLE, both bias and RMSE tend to decrease as $N$ increases, indicating improved finite-sample accuracy. The Wang–Song method does not show the same stable pattern: although its performance often improves with larger $N$, the estimator can be highly sensitive to the added privacy noise, producing occasional large errors that dominate RMSE. This suggests that the proposed method is not only more accurate on average in these experiments, but also more numerically stable.

## Design choices and fairness

The comparison uses the same simulated SAR datasets for both methods within each replicate, with the same true parameters, network model, and privacy budget. The covariates are generated from the bounded distribution used for the Wang–Song comparison setting, so the DGP is not chosen solely to favor DP-QMLE. We vary only the sample size and privacy budget while holding the remaining model parameters fixed, which isolates the finite-sample and privacy effects of interest.

## Limitations and next steps

The current study fixes $p=2$, $\rho=0.2$, one Power-Law network design, and one set of regression coefficients. Therefore, the results should be interpreted as evidence of improved performance in these settings rather than uniform dominance over all SAR models. The Wang–Song estimator also shows occasional extreme estimates, so future work should examine the conditioning of its perturbed linear system and repeat the comparison under additional network structures, dimensions, spatial dependence levels, and privacy budgets. Increasing the number of Monte Carlo replicates would also provide more precise uncertainty estimates.

## Confidence in the results

Confidence is supported by fixed random seeds, paired data generation across methods, saved raw simulation output, automated tests, and Monte Carlo uncertainty bars in the figures. The main factor that could undermine the conclusions is the limited set of DGP conditions; the next priority would be to verify that the same ranking persists across a broader simulation design.
