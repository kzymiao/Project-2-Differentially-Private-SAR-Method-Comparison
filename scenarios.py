"""Central configuration for the DP-SAR method-comparison simulation."""


def make_scenarios():
    """Return the final simulation design as parameter dictionaries.

    The final study crosses an evenly spaced sample-size grid with two privacy
    budgets. All other DGP settings are held fixed so changes in finite-sample
    utility can be attributed to sample size, privacy budget, and method.

    The same N/replicate random streams are intentionally reused across epsilon
    values in the simulation runner. This common-random-number design makes the
    epsilon comparison less noisy while leaving each marginal simulation valid.
    """
    n_values = [400, 600, 800, 1000, 1200]
    epsilon_values = [1.0, 2.0]

    scenarios = []
    for epsilon in epsilon_values:
        for N in n_values:
            scenarios.append(
                {
                    "N": N,
                    "p": 2,
                    "beta": [0.3, 0.3],
                    "rho": 0.2,
                    "network": "PowerLaw",
                    "epsilon": epsilon,
                    "delta": 10 / (N ** 1.1),
                }
            )
    return scenarios
