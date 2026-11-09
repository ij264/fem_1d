import numpy as np
import pytest
from scipy import sparse

import fem_1d as fem


def test_global_shapes_and_types(build):
    space, K, m, _ = build(n=3, Ne=5)
    assert sparse.issparse(K) and K.shape == (space.n_dofs,) * 2
    assert isinstance(m, np.ndarray) and m.shape == (space.n_dofs,)


def test_element_shapes(build):
    n, Ne = 4, 6
    ref = fem.reference_element(n)
    space = fem.FunctionSpace(fem.uniform_mesh(1.0, Ne), ref)
    ones = lambda x: np.ones_like(x)
    assert fem.element_matrix(space, space, ones, 1, 1).shape == (Ne, n + 1, n + 1)
    assert fem.element_load(space, ones).shape == (Ne, n + 1)


def test_stiffness_symmetric(build):
    _, K, _, _ = build(mu=lambda x: 1 + x**2)
    assert abs(K - K.T).max() < 1e-12


def test_element_stiffness_symmetric_and_rows_sum_to_zero():
    ref = fem.reference_element(5)
    space = fem.FunctionSpace(fem.Mesh([0, 0.2, 1.0]), ref)
    Ke = fem.element_matrix(space, space, lambda x: 2 + np.sin(x), 1, 1)
    assert np.allclose(Ke, Ke.transpose(0, 2, 1))
    assert np.allclose(Ke.sum(axis=2), 0, atol=1e-12)


def test_stiffness_positive_semidefinite_with_one_null_vector(build):
    _, K, _, _ = build(n=3, Ne=4, mu=lambda x: 1 + x)
    w = np.linalg.eigvalsh(K.toarray())
    assert w[0] == pytest.approx(0, abs=1e-10)
    assert w[1] > 1e-3


def test_dirichlet_reduced_stiffness_positive_definite(build):
    _, K, _, free = build(n=3, Ne=4)
    Kf = K.toarray()[np.ix_(free, free)]
    assert np.linalg.eigvalsh(Kf)[0] > 0


@pytest.mark.parametrize("n", [2, 3, 5])
def test_energy_of_monomial_is_exact(n, build):
    # int_0^1 (d/dx x^n)^2 dx = n^2 / (2n - 1); GLL is exact for this integrand
    space, K, _, _ = build(n=n, Ne=3)
    u = space.x**n
    assert u @ (K @ u) == pytest.approx(n**2 / (2 * n - 1), rel=1e-12)


def test_energy_scales_with_tension(build):
    space, K1, _, _ = build(mu=lambda x: np.ones_like(x))
    _, K3, _, _ = build(mu=lambda x: 3 * np.ones_like(x))
    u = np.sin(space.x)
    assert u @ (K3 @ u) == pytest.approx(3 * (u @ (K1 @ u)))


def test_stiffness_accepts_scalar_valued_function(build):
    _, K, _, _ = build(mu=lambda x: 2.0)
    _, K2, _, _ = build(mu=lambda x: 2 * np.ones_like(x))
    assert abs(K - K2).max() == 0


def test_energy_with_variable_tension(build):
    # int_0^1 (1 + x) * 1 dx = 1.5 for u = x
    space, K, _, _ = build(mu=lambda x: 1 + x)
    assert space.x @ (K @ space.x) == pytest.approx(1.5)


def test_energy_with_discontinuous_tension(build):
    d = 0.4
    xe = [0, 0.2, d, 0.7, 1.0]
    space, K, _, _ = build(xe=xe, mu=lambda x: np.where(x < d, 1.0, 5.0))
    expected = 1.0 * d + 5.0 * (1 - d)
    assert space.x @ (K @ space.x) == pytest.approx(expected, rel=1e-9)


def test_mass_total_is_integral_of_density(build):
    _, _, m, _ = build(n=3, Ne=2, rho=lambda x: x**5)
    assert m.sum() == pytest.approx(1 / 6)


def test_mass_with_discontinuous_density(build):
    d = 0.5
    _, _, m, _ = build(xe=[0, d, 1], rho=lambda x: np.where(x < d, 2.0, 7.0))
    assert m.sum() == pytest.approx(2 * d + 7 * (1 - d), rel=1e-9)


def test_mass_positive_and_scales_with_length(build):
    _, _, m1, _ = build(a=1.0)
    _, _, m3, _ = build(a=3.0)
    assert np.all(m1 > 0)
    assert np.allclose(m3, 3 * m1)


def test_stiffness_scales_inversely_with_length(build):
    _, K1, _, _ = build(a=1.0)
    _, K2, _, _ = build(a=2.0)
    assert abs(K1 - 2 * K2).max() < 1e-12


def test_shared_node_mass_is_sum_of_both_elements():
    space = fem.FunctionSpace.build(fem.uniform_mesh(1.0, 2), 2)
    one = lambda x: np.ones_like(x)
    me = np.diagonal(fem.element_matrix(space, space, one), axis1=1, axis2=2)
    m = fem.assemble_system(space, [fem.Term(one)])[0].diagonal()
    assert m[2] == pytest.approx(me[0, -1] + me[1, 0])
    assert m[0] == pytest.approx(me[0, 0])


def test_assembly_sums_overlapping_element_entries():
    # degree-1 stiffness of two unit-coefficient elements of length 1/2
    space = fem.FunctionSpace.build(fem.uniform_mesh(1.0, 2), 1)
    A = fem.assemble_system(space, [fem.Term(lambda x: 1.0, 1, 1)])[0].toarray()
    assert np.allclose(A, 2 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]]))


def test_load_assembly_sums_shared_nodes():
    space = fem.FunctionSpace.build(fem.uniform_mesh(3.0, 3), 1)
    assert np.allclose(fem.assemble_load(space, [fem.LoadTerm(lambda x: 1.0)]), [0.5, 1, 1, 0.5])


def test_stiffness_is_banded(build):
    n = 3
    _, K, _, _ = build(n=n, Ne=6)
    coo = K.tocoo()
    assert np.abs(coo.row - coo.col).max() <= n


def test_material_is_sampled_inside_each_element_at_interface():
    # mu jumps exactly at the shared breakpoint; each element must see its own side
    ref = fem.reference_element(2)
    space = fem.FunctionSpace(fem.Mesh([0, 0.5, 1.0]), ref)
    Ke = fem.element_matrix(space, space, lambda x: np.where(x < 0.5, 1.0, 10.0), 1, 1)
    K_left = fem.element_matrix(space, space, lambda x: np.ones_like(x), 1, 1)[0]
    assert np.allclose(Ke[0], K_left)
    assert np.allclose(Ke[1], 10 * K_left)
