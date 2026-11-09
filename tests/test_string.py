"""Assembly checked against the exact spectra of strings, -(mu u')' = w^2 rho u."""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq

import fem_1d as fem


def test_stiffness_annihilates_constants(build):
    _, K, _, _ = build()
    assert np.abs(K @ np.ones(K.shape[0])).max() < 1e-10


def test_lumped_mass_integrates_density(build):
    _, _, m, _ = build(rho=lambda x: 1 + x)
    assert m.sum() == pytest.approx(1.5)


@pytest.mark.parametrize("n, Ne", [(2, 10), (4, 10), (6, 4)])
def test_low_frequencies_match_fixed_string(build, modes, n, Ne):
    omega, _ = modes(*build(n=n, Ne=Ne)[1:], k=3)
    assert np.allclose(omega, np.pi * np.arange(1, 4), rtol=1e-3)


def test_nonuniform_mesh_frequencies(build, modes):
    xe = np.sort(np.r_[0.0, np.random.default_rng(0).uniform(0.05, 0.95, 9), 1.0])
    omega, _ = modes(*build(n=6, xe=xe)[1:], k=3)
    assert np.allclose(omega, np.pi * np.arange(1, 4), rtol=1e-6)


@pytest.mark.parametrize("a, mu, rho", [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 4.0, 1.0), (3.0, 2.0, 8.0)])
def test_uniform_string_frequencies(build, modes, a, mu, rho):
    _, K, m, free = build(n=6, Ne=6, a=a, mu=lambda x: mu * np.ones_like(x), rho=lambda x: rho * np.ones_like(x))
    omega, _ = modes(K, m, free, k=3)
    assert np.allclose(omega, np.sqrt(mu / rho) * np.pi * np.arange(1, 4) / a, rtol=1e-6)


def test_fixed_free_string_frequencies(build, modes):
    _, K, m, free = build(n=6, Ne=6)
    free[-1] = True  # right end free: zero tension force there, nothing to add
    omega, _ = modes(K, m, free, k=3)
    assert np.allclose(omega, np.pi * (np.arange(3) + 0.5), rtol=1e-6)


def test_modes_match_generalised_eigh_and_are_mass_orthonormal(build, modes):
    _, K, m, free = build(mu=lambda x: 1 + x**2, rho=lambda x: 2 - x)
    omega, phi = modes(K, m, free)
    w2 = eigh(K.toarray()[np.ix_(free, free)], np.diag(m[free]), eigvals_only=True)
    assert np.allclose(omega, np.sqrt(w2))
    assert np.allclose(phi.T @ (m[:, None] * phi), np.eye(len(omega)), atol=1e-10)


def test_mode_p_has_p_minus_one_interior_sign_changes(build, modes):
    _, phi = modes(*build(n=4, Ne=10)[1:], k=5)
    for p in range(5):
        v = phi[1:-1, p]
        assert np.sum(v[:-1] * v[1:] < 0) == p


def test_frequency_convergence_in_h_is_order_2n(build, modes):
    n = 2
    errs = [abs(modes(*build(n=n, Ne=Ne)[1:], k=1)[0][0] - np.pi) for Ne in (4, 8)]
    assert np.log2(errs[0] / errs[1]) > 2 * n - 0.5


def test_frequency_convergence_in_p_is_spectral(build, modes):
    errs = [abs(modes(*build(n=n, Ne=2)[1:], k=1)[0][0] - np.pi) for n in (2, 3, 4, 5, 6)]
    assert errs[-1] < 1e-8 and all(b < a for a, b in zip(errs, errs[1:]))


def test_two_layer_matches_transcendental_frequencies(build, modes):
    # Fixed ends, mu jumps from mu1 to mu2 at d (rho = 1). With k_i = omega / c_i the
    # eigenfrequencies are the roots of
    #   mu1 k1 cos(k1 d) sin(k2 (a - d)) + mu2 k2 sin(k1 d) cos(k2 (a - d)) = 0.
    a, d, mu1, mu2 = 1.0, 0.5, 1.0, 4.0
    c1, c2 = np.sqrt(mu1), np.sqrt(mu2)

    def f(w):
        k1, k2 = w / c1, w / c2
        return mu1 * k1 * np.cos(k1 * d) * np.sin(k2 * (a - d)) + mu2 * k2 * np.sin(k1 * d) * np.cos(k2 * (a - d))

    grid = np.linspace(0.1, 15, 3000)
    vals = f(grid)
    roots = [brentq(f, lo, hi) for lo, hi, flo, fhi in zip(grid, grid[1:], vals, vals[1:]) if flo * fhi < 0]

    xe = fem.sized_mesh([0.0, d, a], drmax=0.1).xe
    omega, _ = modes(*build(n=6, xe=xe, mu=lambda x: np.where(x < d, mu1, mu2))[1:], k=3)
    assert np.allclose(omega, roots[:3], rtol=1e-8)
