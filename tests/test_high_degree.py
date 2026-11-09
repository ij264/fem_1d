"""GLL nodes, weights, derivative matrix and barycentric basis at high degree."""

import numpy as np
import pytest
from scipy.special import eval_legendre

import fem_1d as fem
from fem_1d.gll import basis_at

HIGH = [16, 32, 48, 64]


@pytest.mark.parametrize("n", HIGH)
def test_quadrature_exact_on_legendre_polynomials(n):
    # int P_k = 0 for 1 <= k <= 2n - 1; Legendre polynomials keep this well conditioned
    ref = fem.reference_element(n)
    assert ref.weights.sum() == pytest.approx(2.0, abs=1e-14)
    err = max(abs(ref.weights @ eval_legendre(k, ref.xi)) for k in range(1, 2 * n))
    assert err < 1e-14


@pytest.mark.parametrize("n", HIGH)
def test_interior_nodes_are_roots_of_legendre_derivative(n):
    ref = fem.reference_element(n)
    dP = np.polynomial.legendre.Legendre.basis(n).deriv()
    assert np.abs(dP(ref.xi[1:-1])).max() < 1e-10 * n**2


@pytest.mark.parametrize("n", HIGH)
def test_derivative_matrix_rows_sum_to_zero_to_rounding(n):
    assert np.abs(fem.reference_element(n).D.sum(axis=1)).max() < 1e-12


@pytest.mark.parametrize("n", HIGH)
def test_derivative_matrix_accurate_on_random_polynomials(n):
    ref = fem.reference_element(n)
    for seed in range(3):
        p = np.polynomial.Polynomial(np.random.default_rng(seed).standard_normal(n + 1))
        ex = p.deriv()(ref.xi)
        assert np.abs(ref.D @ p(ref.xi) - ex).max() < 1e-12 * np.abs(ex).max()


@pytest.mark.parametrize("n", [4, 12, 24, 40])
def test_basis_at_higher_rule_accurate(n):
    ref, quad = fem.reference_element(n), fem.reference_element(n + 3)
    B, Dxi = basis_at(ref, quad)
    p = np.polynomial.Polynomial(np.random.default_rng(0).standard_normal(n + 1))
    assert np.abs(B @ p(ref.xi) - p(quad.xi)).max() < 1e-13 * np.abs(p(quad.xi)).max()
    assert np.abs(Dxi @ p(ref.xi) - p.deriv()(quad.xi)).max() < 1e-12 * np.abs(p.deriv()(quad.xi)).max()


def test_corner_entries_keep_their_analytic_values():
    for n in (2, 8, 30):
        D = fem.reference_element(n).D
        assert D[0, 0] == pytest.approx(-n * (n + 1) / 4, rel=1e-12)
        assert D[-1, -1] == pytest.approx(n * (n + 1) / 4, rel=1e-12)
        assert np.abs(np.diag(D)[1:-1]).max() < 1e-10 * n**2


# --- lagrange_basis -------------------------------------------------------


def test_lagrange_basis_is_kronecker_on_nodes():
    nodes = fem.reference_element(7).xi
    assert np.array_equal(fem.lagrange_basis(nodes, nodes), np.eye(8))


def test_lagrange_basis_partition_of_unity_and_exactness():
    nodes = fem.reference_element(9).xi
    x = np.linspace(-1, 1, 101)
    L = fem.lagrange_basis(nodes, x)
    assert np.allclose(L.sum(axis=1), 1, atol=1e-14)
    p = np.polynomial.Polynomial([1, -2, 0.5, 3, 0, 0, 1, 0, 0, -1])
    assert np.allclose(L @ p(nodes), p(x), atol=1e-12)


def test_lagrange_basis_near_a_node():
    nodes = fem.reference_element(5).xi
    L = fem.lagrange_basis(nodes, [nodes[2] + 1e-17, nodes[2] + 1e-9])
    assert np.allclose(L[0], np.eye(6)[2]) and np.allclose(L[1], np.eye(6)[2], atol=1e-7)
    assert np.all(np.isfinite(L))


def test_lagrange_basis_scalar_point():
    assert fem.lagrange_basis([-1.0, 1.0], 0.0).shape == (1, 2)


# --- assembly at high degree ----------------------------------------------


def test_high_degree_stiffness_annihilates_constants():
    s = fem.FunctionSpace.build(fem.uniform_mesh(1.0, 3), 40)
    K, _ = fem.assemble_system(s, [fem.Term(lambda x: 1 + x, 1, 1)])
    assert np.abs(K @ np.ones(s.n_dofs)).max() < 1e-10


def test_high_degree_string_frequencies(build, modes):
    omega, _ = modes(*build(n=32, Ne=2)[1:], k=10)
    assert np.allclose(omega, np.pi * np.arange(1, 11), rtol=1e-12)
