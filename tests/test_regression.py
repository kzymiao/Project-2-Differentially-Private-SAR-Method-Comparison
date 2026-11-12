import numpy as np
import random

from benchmarks.baseline.function_qmle_dp import (
    qmle_dp_estimate as baseline_qmle_dp_estimate,
)

from src.function_qmle_dp import (
    qmle_dp_estimate as optimized_qmle_dp_estimate,
)

from src.generation import generate_compare


def test_linear_solve_matches_explicit_inverse():
    rng = np.random.default_rng(123)

    N = 30
    rho = 0.2

    # Construct a fixed row-normalized weight matrix.
    W = rng.uniform(size=(N, N))
    np.fill_diagonal(W, 0.0)
    W = W / W.sum(axis=1, keepdims=True)

    S = np.eye(N) - rho * W

    rhs = rng.normal(size=(N, 1))

    # Unit 2 implementation
    y_baseline = np.linalg.inv(S).dot(rhs)

    # Unit 3 optimized implementation
    y_optimized = np.linalg.solve(S, rhs)

    np.testing.assert_allclose(
        y_optimized,
        y_baseline,
        rtol=1e-8,
        atol=1e-10,
    )


def test_trace_identity_matches_full_matrix_product():
    rng = np.random.default_rng(123)

    N = 30
    rho = 0.2

    W = rng.uniform(
        size=(N, N)
    )

    S = (
        np.eye(N)
        - rho * W
    )

    S_inv = np.linalg.inv(S)

    baseline = np.trace(
        W.dot(S_inv)
    )

    optimized = np.sum(
        W * S_inv.T
    )

    np.testing.assert_allclose(
        optimized,
        baseline,
        rtol=1e-12,
        atol=1e-12,
    )


def test_optimized_qmle_matches_baseline():
    N = 80
    p = 2

    beta = np.array(
        [[0.3], [0.3]]
    )

    rho = 0.2

    np.random.seed(123)
    random.seed(123)

    X, Y, W = generate_compare(
        N,
        p,
        beta,
        rho,
        "PowerLaw",
    )

    epsilon = 1.0
    delta = 10 / (N ** 1.1)

    # Use exactly the same DP noise stream.
    np.random.seed(999)
    random.seed(999)

    baseline, *_ = (
        baseline_qmle_dp_estimate(
            X.copy(),
            Y.copy(),
            W.copy(),
            epsilon,
            delta,
        )
    )

    np.random.seed(999)
    random.seed(999)

    optimized, *_ = (
        optimized_qmle_dp_estimate(
            X.copy(),
            Y.copy(),
            W.copy(),
            epsilon,
            delta,
        )
    )

    np.testing.assert_allclose(
        optimized,
        baseline,
        rtol=1e-6,
        atol=1e-8,
    )