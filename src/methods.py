"""Wrappers for the two private SAR estimators compared in the study."""

import numpy as np

from .function_qmle_dp import qmle_dp_estimate
from .function_wangsong import lse_dp_estimate, lse_dp_gaussian_estimate


class DPQMLE:
    """Proposed iterative gradient-perturbation DP-QMLE."""

    name = "DP-QMLE (ours)"

    def fit(self, data, settings):
        estimate, *_ = qmle_dp_estimate(
            data["X"], data["Y"], data["W"], settings["epsilon"], settings["delta"]
        )
        return np.asarray(estimate).reshape(-1)


class WangSongGaussian:
    """Gaussianized Wang--Song functional mechanism matched to (epsilon, delta)-DP."""

    name = "Wang-Song Gaussian FM"

    def fit(self, data, settings):
        c3 = 0.8 * np.log(settings["N"])
        estimate, *_ = lse_dp_gaussian_estimate(
            data["X"], data["Y"], data["W"],
            epsilon=settings["epsilon"], delta=settings["delta"], c3=c3,
        )
        return np.asarray(estimate).reshape(-1)


class WangSongLaplace:
    """Original Wang--Song pure-epsilon functional-mechanism baseline."""

    name = "Wang-Song Laplace FM"

    def fit(self, data, settings):
        c3 = 0.8 * np.log(settings["N"])
        estimate, *_ = lse_dp_estimate(
            data["X"], data["Y"], data["W"], epsilon=settings["epsilon"], c3=c3
        )
        return np.asarray(estimate).reshape(-1)


PRIMARY_METHODS = [DPQMLE(), WangSongGaussian()]
