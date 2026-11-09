import dataclasses

import numpy as np
import pytest

import fem_1d as fem

DEGREES = [1, 2, 3, 4, 5, 8, 12]


@pytest.mark.parametrize("n", DEGREES)
def test_shapes_and_endpoints(n):
    ref = fem.reference_element(n)
    assert ref.n == n
    assert ref.xi.shape == ref.weights.shape == (n + 1,)
    assert ref.D.shape == (n + 1, n + 1)
    assert ref.xi[0] == -1.0 and ref.xi[-1] == 1.0


@pytest.mark.parametrize("n", DEGREES)
def test_nodes_sorted_and_symmetric(n):
    xi = fem.reference_element(n).xi
    assert np.all(np.diff(xi) > 0)
    assert np.allclose(xi, -xi[::-1], atol=1e-14)


@pytest.mark.parametrize("n", DEGREES)
def test_weights_positive_and_symmetric(n):
    w = fem.reference_element(n).weights
    assert np.all(w > 0)
    assert np.allclose(w, w[::-1])
    assert w.sum() == pytest.approx(2.0)


def test_known_low_order_rules():
    r1 = fem.reference_element(1)
    assert np.allclose(r1.xi, [-1, 1]) and np.allclose(r1.weights, [1, 1])
    r2 = fem.reference_element(2)
    assert np.allclose(r2.xi, [-1, 0, 1])
    assert np.allclose(r2.weights, [1 / 3, 4 / 3, 1 / 3])
    r3 = fem.reference_element(3)
    assert np.allclose(r3.xi, [-1, -1 / np.sqrt(5), 1 / np.sqrt(5), 1])
    assert np.allclose(r3.weights, [1 / 6, 5 / 6, 5 / 6, 1 / 6])


@pytest.mark.parametrize("n", DEGREES)
def test_quadrature_exact_to_degree_2n_minus_1(n):
    ref = fem.reference_element(n)
    for p in range(2 * n):
        exact = 0.0 if p % 2 else 2.0 / (p + 1)
        assert ref.weights @ ref.xi**p == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_quadrature_not_exact_at_degree_2n(n):
    ref = fem.reference_element(n)
    assert abs(ref.weights @ ref.xi ** (2 * n) - 2.0 / (2 * n + 1)) > 1e-4


@pytest.mark.parametrize("n", DEGREES)
def test_derivative_matrix_exact_on_polynomials_up_to_degree_n(n):
    ref = fem.reference_element(n)
    for p in range(n + 1):
        f = ref.xi**p
        df = p * ref.xi ** (p - 1) if p else np.zeros_like(ref.xi)
        assert np.allclose(ref.D @ f, df, atol=1e-9 * 10 ** (n / 4))


@pytest.mark.parametrize("n", DEGREES)
def test_derivative_matrix_rows_sum_to_zero(n):
    assert np.allclose(fem.reference_element(n).D.sum(axis=1), 0, atol=1e-10)


@pytest.mark.parametrize("n", DEGREES)
def test_derivative_matrix_centro_antisymmetry(n):
    D = fem.reference_element(n).D
    assert np.allclose(D, -D[::-1, ::-1], atol=1e-10)


@pytest.mark.parametrize("n", DEGREES)
def test_summation_by_parts(n):
    # W D + (W D)^T = diag(-1, 0, ..., 0, 1): discrete integration by parts
    ref = fem.reference_element(n)
    S = ref.weights[:, None] * ref.D
    B = np.zeros((n + 1, n + 1))
    B[0, 0], B[-1, -1] = -1.0, 1.0
    assert np.allclose(S + S.T, B, atol=1e-10)


def test_derivative_matrix_matches_lagrange_polynomials():
    n = 4
    ref = fem.reference_element(n)
    for k in range(n + 1):
        e = np.zeros(n + 1)
        e[k] = 1.0
        lk = np.polynomial.Polynomial.fit(ref.xi, e, n)
        assert np.allclose(ref.D[:, k], lk.deriv()(ref.xi), atol=1e-9)


def test_corner_entries():
    n = 6
    D = fem.reference_element(n).D
    assert D[0, 0] == pytest.approx(-n * (n + 1) / 4)
    assert D[-1, -1] == pytest.approx(n * (n + 1) / 4)


@pytest.mark.parametrize("n", [0, -1, -5])
def test_invalid_degree_raises(n):
    with pytest.raises(ValueError):
        fem.reference_element(n)


def test_reference_element_is_frozen():
    ref = fem.reference_element(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.n = 4
