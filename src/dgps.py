"""Data-generating process for the SAR comparison study."""

import random
import numpy as np

from .generation import generate_compare


def generate_sar_data(settings, seed):
    """Generate one SAR dataset under the Wang--Song-compatible DGP.

    The original generator uses NumPy's global RNG and Python's ``random``
    module, so both are seeded here to make each replicate reproducible.
    """
    np.random.seed(seed)
    random.seed(seed)
    beta = np.asarray(settings["beta"], dtype=float).reshape(settings["p"], 1)
    X, Y, W = generate_compare(
        settings["N"], settings["p"], beta, settings["rho"], settings["network"]
    )
    return {"X": X, "Y": Y, "W": W, "beta_true": beta.flatten(), "rho_true": settings["rho"]}
