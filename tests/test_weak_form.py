import numpy as np
import pytest

import fem_1d as fem
from fem_1d.gll import basis_at

pi = np.pi


def space(n, Ne, xe=None):
    return fem.FunctionSpace.build(np.linspace(0, 1, Ne + 1) if xe is None else xe, n)


def solve(s, terms, loads, left=0.0, right=0.0):
    """Solve with Dirichlet values at both ends; returns the dof values."""
    bcs = [fem.Dirichlet("u", "left", left), fem.Dirichlet("u", "right", right)]
    return fem.Problem(s, terms, loads, bcs).solve().u


# --- FunctionSpace / basis ------------------------------------------------


def test_function_space_properties():
    s = space(3, 4)
    assert s.n_dofs == 13 and s.x[0] == 0 and s.x[-1] == pytest.approx(1)
    assert s.conn.shape == (4, 4)


def test_spaces_of_different_degree_share_one_mesh():
    mesh = fem.uniform_mesh(1.0, 3)
    u = fem.FunctionSpace(mesh, fem.reference_element(4))
    p = fem.FunctionSpace(mesh, fem.reference_element(2))
    assert u.mesh is p.mesh and (u.n_dofs, p.n_dofs) == (13, 7)


def test_function_space_accepts_end_points_instead_of_mesh():
    s = fem.FunctionSpace([0.0, 0.5, 1.0], fem.reference_element(2))
    assert isinstance(s.mesh, fem.Mesh) and s.n_dofs == 5


@pytest.mark.parametrize("n", [1, 3, 6])
def test_basis_on_own_nodes_is_identity(n):
    ref = fem.reference_element(n)
    B, D = basis_at(ref, ref)
    assert np.array_equal(B, np.eye(n + 1)) and np.array_equal(D, ref.D)


@pytest.mark.parametrize("n, nq", [(1, 5), (2, 6), (4, 7)])
def test_basis_at_other_nodes_interpolates_polynomials(n, nq):
    ref, quad = fem.reference_element(n), fem.reference_element(nq)
    B, D = basis_at(ref, quad)
    f = lambda x: 1 + 2 * x - x**n
    df = lambda x: 2 - n * x ** (n - 1)
    assert np.allclose(B @ f(ref.xi), f(quad.xi), atol=1e-12)
    assert np.allclose(D @ f(ref.xi), df(quad.xi), atol=1e-11)
    assert np.allclose(B.sum(axis=1), 1)  # partition of unity


# --- assembly -------------------------------------------------------------


def test_reaction_term_on_own_nodes_is_diagonal_lumped_mass():
    s = space(4, 5)
    rho = lambda x: 2 - x
    M, _ = fem.assemble_system(s, [fem.Term(rho)])
    assert abs(M - __import__("scipy").sparse.diags_array(M.diagonal())).max() == 0
    w = np.zeros(s.n_dofs)
    np.add.at(w, s.conn, s.ref.weights * (s.mesh.he / 2)[:, None])
    assert np.allclose(M.diagonal(), w * rho(s.x))


def test_terms_are_additive():
    s = space(3, 4)
    one = lambda x: np.ones_like(x)
    A1, _ = fem.assemble_system(s, [fem.Term(one, 1, 1)])
    A2, _ = fem.assemble_system(s, [fem.Term(one)])
    A12, _ = fem.assemble_system(s, [fem.Term(one, 1, 1), fem.Term(one)])
    assert abs(A12 - A1 - A2).max() < 1e-14


def test_advection_term_is_nonsymmetric_and_rows_of_constants_sum_correctly():
    s = space(4, 3)
    C, _ = fem.assemble_system(s, [fem.Term(lambda x: 1.0, d_test=0, d_trial=1)])
    assert abs(C - C.T).max() > 0.1
    assert np.abs(C @ np.ones(s.n_dofs)).max() < 1e-12  # derivative of a constant
    assert np.ones(s.n_dofs) @ (C @ s.x) == pytest.approx(1.0)  # int u' = 1


def test_mixed_degree_coupling_is_exact_for_polynomials():
    # int q u' dx with u = x^2 (degree 3 space), q = x (degree 1 space) = 2/3
    spaces = {"u": space(3, 3), "q": space(1, 3)}
    A, sl = fem.assemble_system(spaces, [fem.Term(lambda x: 1.0, 0, 1, test="q", trial="u")])
    C = A[sl["q"], sl["u"]]
    assert C.shape == (spaces["q"].n_dofs, spaces["u"].n_dofs)
    assert spaces["q"].x @ (C @ spaces["u"].x**2) == pytest.approx(2 / 3)


def test_mixed_degree_with_variable_coefficient():
    spaces = {"u": space(4, 2), "p": space(2, 2)}
    c = lambda x: 1 + x
    A, sl = fem.assemble_system(spaces, [fem.Term(c, 0, 1, test="p", trial="u")])
    C = A[sl["p"], sl["u"]]
    # int (1 + x) * p * u' with p = 1, u = x^3  ->  int (1 + x) 3 x^2 = 1 + 3/4
    p = np.ones(spaces["p"].n_dofs)
    assert p @ (C @ spaces["u"].x**3) == pytest.approx(1.75)


def test_transposed_terms_give_transposed_blocks():
    spaces = {"u": space(4, 3), "p": space(2, 3)}
    c = lambda x: 1 + x**2
    A, sl = fem.assemble_system(spaces, [
        fem.Term(c, 0, 1, test="p", trial="u"),
        fem.Term(c, 1, 0, test="u", trial="p"),
    ])
    assert abs(A[sl["p"], sl["u"]] - A[sl["u"], sl["p"]].T).max() < 1e-13


def test_block_layout_and_slices():
    spaces = {"u": space(3, 2), "p": space(1, 2)}
    sl = fem.dof_slices(spaces)
    assert sl["u"] == slice(0, 7) and sl["p"] == slice(7, 10)
    A, _ = fem.assemble_system(spaces, [fem.Term(lambda x: 1.0, 1, 1, test="p", trial="p")])
    assert A.shape == (10, 10) and abs(A[:7]).max() == 0 and abs(A[:, :7]).max() == 0


def test_block_diagonal_fields_are_independent():
    spaces = {"u": space(3, 3), "v": space(3, 3)}
    A, sl = fem.assemble_system(spaces, [
        fem.Term(lambda x: 1.0, 1, 1, test="u", trial="u"),
        fem.Term(lambda x: 4.0, 1, 1, test="v", trial="v"),
    ])
    assert abs(A[sl["v"], sl["v"]] - 4 * A[sl["u"], sl["u"]]).max() < 1e-12
    assert abs(A[sl["u"], sl["v"]]).max() == 0


def test_empty_term_list_gives_zero_matrix():
    A, _ = fem.assemble_system(space(2, 2), [])
    assert A.shape == (5, 5) and A.nnz == 0


def test_fields_must_share_element_endpoints():
    with pytest.raises(ValueError):
        fem.assemble_system({"u": space(2, 3), "v": space(2, 4)}, [])


def test_unknown_field_raises():
    with pytest.raises(KeyError):
        fem.assemble_system(space(2, 2), [fem.Term(lambda x: 1.0, test="nope")])


def test_second_derivatives_rejected():
    with pytest.raises(ValueError):
        fem.assemble_system(space(2, 2), [fem.Term(lambda x: 1.0, 2, 0)])


def test_discontinuous_coefficient_is_sampled_per_element():
    s = space(3, 2, xe=[0, 0.5, 1.0])
    c = lambda x: np.where(x < 0.5, 1.0, 9.0)
    A, _ = fem.assemble_system(s, [fem.Term(c)])
    assert A.diagonal().sum() == pytest.approx(0.5 + 9 * 0.5)


# --- load vectors ---------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5])
def test_load_integrates_against_test_function(n):
    s = space(n, 3)
    b = fem.assemble_load(s, [fem.LoadTerm(lambda x: x)])
    assert b.sum() == pytest.approx(0.5)  # basis functions sum to one


def test_load_with_derivative_test_function():
    s = space(3, 3)
    b = fem.assemble_load(s, [fem.LoadTerm(lambda x: 2.0, d_test=1)])
    # int 2 v' dx = 2 (v(1) - v(0)) -> picks out the end nodes
    assert b @ s.x == pytest.approx(2.0)  # v = x
    assert b @ np.ones(s.n_dofs) == pytest.approx(0, abs=1e-12)


def test_load_in_multi_field_lands_in_right_block():
    spaces = {"u": space(2, 2), "p": space(1, 2)}
    b = fem.assemble_load(spaces, [fem.LoadTerm(lambda x: 1.0, test="p")])
    sl = fem.dof_slices(spaces)
    assert np.all(b[sl["u"]] == 0) and b[sl["p"]].sum() == pytest.approx(1.0)


# --- manufactured solutions ----------------------------------------------


def test_poisson_nonzero_dirichlet_exact_for_quadratic():
    s = space(2, 3)
    one = lambda x: 1.0
    u = solve(s, [fem.Term(one, 1, 1)], [fem.LoadTerm(lambda x: -2.0)], left=1.0, right=2.0)
    assert np.allclose(u, s.x**2 + 1, atol=1e-12)


def test_natural_neumann_condition():
    # -u'' = -2, u(0) = 0, u'(1) = 2  ->  u = x^2; flux enters the load at the free end
    s = space(2, 3)
    loads = [fem.LoadTerm(lambda x: -2.0), fem.PointLoad(2.0, "right")]
    u = fem.Problem(s, [fem.Term(lambda x: 1.0, 1, 1)], loads, [fem.Dirichlet("u", "left")]).solve().u
    assert np.allclose(u, s.x**2, atol=1e-12)


def test_flux_continuity_across_discontinuous_tension():
    # -(mu u')' = 0, u(0) = 0, u(1) = 1: u is piecewise linear with mu u' constant
    d, mu1, mu2 = 0.3, 1.0, 5.0
    s = space(3, 4, xe=[0, 0.15, d, 0.6, 1.0])
    mu = lambda x: np.where(x < d, mu1, mu2)
    u = solve(s, [fem.Term(mu, 1, 1)], [], left=0.0, right=1.0)
    q = 1.0 / (d / mu1 + (1 - d) / mu2)
    expected = np.where(s.x < d, q * s.x / mu1, q * d / mu1 + q * (s.x - d) / mu2)
    assert np.allclose(u, expected, atol=1e-12)


def err_poisson(n, Ne):
    s = space(n, Ne)
    mu = lambda x: 1 + x
    f = lambda x: -pi * np.cos(pi * x) + (1 + x) * pi**2 * np.sin(pi * x)
    u = solve(s, [fem.Term(mu, 1, 1)], [fem.LoadTerm(f)])
    return np.abs(u - np.sin(pi * s.x)).max()


def test_variable_coefficient_poisson_spectral_in_p():
    errs = [err_poisson(n, 3) for n in (2, 4, 6, 8)]
    assert all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 1e-7


def test_variable_coefficient_poisson_algebraic_in_h():
    e1, e2 = err_poisson(2, 8), err_poisson(2, 16)
    assert np.log2(e1 / e2) > 1.8


def test_reaction_diffusion_converges():
    # -u'' + 5 u = f, u = sin(pi x)
    errs = []
    for n in (4, 7):
        s = space(n, 3)
        f = lambda x: (pi**2 + 5) * np.sin(pi * x)
        u = solve(s, [fem.Term(lambda x: 1.0, 1, 1), fem.Term(lambda x: 5.0)], [fem.LoadTerm(f)])
        errs.append(np.abs(u - np.sin(pi * s.x)).max())
    assert errs[1] < errs[0] and errs[1] < 1e-6


def test_advection_diffusion_converges_nonsymmetric():
    # -u'' + 2 u' = f, u = sin(pi x): the test function multiplies u'
    errs = []
    for n in (4, 7):
        s = space(n, 3)
        f = lambda x: pi**2 * np.sin(pi * x) + 2 * pi * np.cos(pi * x)
        terms = [fem.Term(lambda x: 1.0, 1, 1), fem.Term(lambda x: 2.0, 0, 1)]
        u = solve(s, terms, [fem.LoadTerm(f)])
        errs.append(np.abs(u - np.sin(pi * s.x)).max())
    assert errs[1] < errs[0] and errs[1] < 1e-6


def mixed_poisson(n_u, n_s, Ne=3):
    # sigma = u', -sigma' = f, weakly:
    #   (sigma, tau) - (u', tau) = 0,   (sigma, v') = (f, v),   u = 0 at both ends
    spaces = {"u": space(n_u, Ne), "s": space(n_s, Ne)}
    terms = [
        fem.Term(lambda x: 1.0, 0, 0, test="s", trial="s"),
        fem.Term(lambda x: -1.0, 0, 1, test="s", trial="u"),
        fem.Term(lambda x: 1.0, 1, 0, test="u", trial="s"),
    ]
    f = lambda x: pi**2 * np.sin(pi * x)
    bcs = [fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")]
    return spaces, fem.Problem(spaces, terms, [fem.LoadTerm(f, test="u")], bcs)


def test_mixed_poisson_first_order_system():
    spaces, p = mixed_poisson(5, 5)
    sol = p.solve()
    assert np.abs(sol["u"] - np.sin(pi * spaces["u"].x)).max() < 1e-4
    assert np.abs(sol["s"] - pi * np.cos(pi * spaces["s"].x)).max() < 1e-3


def test_mixed_poisson_with_too_small_sigma_space_is_singular():
    # sigma of degree n - 2 cannot control u' (an inf-sup failure, as for Stokes pressure)
    _, p = mixed_poisson(5, 3)
    Af = p.A.toarray()[np.ix_(p.free, p.free)]
    assert np.linalg.matrix_rank(Af) < Af.shape[0]
