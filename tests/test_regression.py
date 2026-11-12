import numpy as np


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
