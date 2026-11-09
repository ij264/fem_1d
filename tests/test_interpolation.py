"""Evaluating fields between nodes, and point functionals for kernels."""

import numpy as np
import pytest

import fem_1d as fem

one = lambda x: 1.0


@pytest.mark.parametrize("n", [1, 3, 6])
def test_interpolation_exact_for_polynomials_of_space_degree(n):
    s = fem.FunctionSpace.build(fem.Mesh([0.0, 0.3, 0.7, 1.2]), n)
    p = np.polynomial.Polynomial(np.arange(1.0, n + 2))
    x = np.linspace(0, 1.2, 37)
    assert np.allclose(s.evaluate(p(s.x), x), p(x), atol=1e-12)
    assert np.allclose(s.evaluate(p(s.x), x, d=1), p.deriv()(x), atol=1e-10)


def test_interpolation_at_nodes_returns_nodal_values():
    s = fem.FunctionSpace.build(fem.uniform_mesh(1.0, 3), 4)
    u = np.random.default_rng(0).standard_normal(s.n_dofs)
    assert np.allclose(s.evaluate(u, s.x), u, atol=1e-14)


def test_interpolation_matrix_rows_sum_to_one_and_derivative_rows_to_zero():
    s = fem.FunctionSpace.build(fem.uniform_mesh(1.0, 4), 3)
    x = np.linspace(0, 1, 23)
    assert np.allclose(s.interpolation_matrix(x).sum(axis=1), 1)
    assert np.allclose(s.interpolation_matrix(x, d=1).sum(axis=1), 0, atol=1e-11)
    assert s.interpolation_matrix(x).nnz == 23 * 4


def test_derivative_jumps_between_elements_and_side_selects_it():
    # piecewise-linear hat: slope +2 below 0.5, -2 above
    s = fem.FunctionSpace.build([0.0, 0.5, 1.0], 1)
    u = np.array([0.0, 1.0, 0.0])
    assert s.evaluate(u, 0.5, d=1, side="below")[0] == pytest.approx(2.0)
    assert s.evaluate(u, 0.5, d=1, side="above")[0] == pytest.approx(-2.0)
    assert s.evaluate(u, 0.5, side="below")[0] == s.evaluate(u, 0.5, side="above")[0] == 1.0


def test_discontinuous_field_side_picks_its_own_dof():
    s = fem.FunctionSpace.build([0.0, 0.5, 1.0], 2, discontinuous=True)
    u = np.where(np.arange(s.n_dofs) < 3, 1.0, 7.0)  # element 0 owns dofs 0-2
    assert s.evaluate(u, 0.5, side="below")[0] == pytest.approx(1.0)
    assert s.evaluate(u, 0.5, side="above")[0] == pytest.approx(7.0)


def test_domain_ends():
    s = fem.FunctionSpace.build([2.0, 3.0], 3)
    u = s.x**2
    assert s.evaluate(u, [2.0, 3.0]) == pytest.approx([4.0, 9.0])
    assert s.evaluate(u, 3.0, side="above")[0] == pytest.approx(9.0)


@pytest.mark.parametrize("x, d", [(-0.5, 0), (1.5, 0), (0.5, 2)])
def test_interpolation_errors(x, d):
    with pytest.raises(ValueError):
        fem.FunctionSpace.build([0.0, 1.0], 2).evaluate(np.zeros(3), x, d=d)


# --- solutions and functionals ----------------------------------------------


def poisson(n=4, Ne=4, params=None):
    s = fem.FunctionSpace.build(fem.uniform_mesh(1.0, Ne), n)
    return fem.Problem(s, [fem.Term(one, 1, 1, param="mu")], [fem.LoadTerm(lambda x: np.pi**2 * np.sin(np.pi * x))],
                       [fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")],
                       params=params or {"mu": 1.0})


def test_solution_evaluate_between_nodes_is_spectrally_accurate():
    sol = poisson(n=8, Ne=3).solve()
    x = np.linspace(0, 1, 41)
    assert np.abs(sol.evaluate("u", x) - np.sin(np.pi * x)).max() < 1e-7
    assert np.abs(sol.evaluate("u", x, d=1) - np.pi * np.cos(np.pi * x)).max() < 1e-5


def test_functional_matches_evaluation():
    p = poisson()
    sol = p.solve()
    for x, d in [(0.123, 0), (0.5, 1), (0.77, 1)]:
        assert p.functional("u", x, d) @ sol.u == pytest.approx(sol.evaluate("u", x, d)[0])


def test_functional_at_node_picks_out_that_dof():
    p = poisson()
    for at in (0.25, "left", "right"):
        e = np.zeros(p.n_dofs)
        e[p.dof("u", at)] = 1.0
        assert np.allclose(p.functional("u", at), e)


def test_kernel_of_off_node_value_and_derivative_match_finite_difference():
    params = {"mu": lambda x: 1 + x}
    p = poisson(params=params)
    dm = lambda x: np.cos(2 * x)
    eps = 1e-6
    for x0, d in [(0.37, 0), (0.61, 1)]:
        c = p.functional("u", x0, d)
        K = p.kernel("mu", c=c)
        J = lambda m: c @ poisson(params={"mu": m}).solve().u
        fd = (J(lambda x: params["mu"](x) + eps * dm(x)) - J(lambda x: params["mu"](x) - eps * dm(x))) / (2 * eps)
        assert K.apply(dm) == pytest.approx(fd, rel=1e-6)


def test_functional_in_multi_field_problem_lands_in_field_block():
    xe = fem.uniform_mesh(1.0, 3)
    spaces = {"a": fem.FunctionSpace.build(xe, 2), "b": fem.FunctionSpace.build(xe, 3)}
    p = fem.Problem(spaces, [fem.Term(one, 1, 1, f, f) for f in spaces])
    c = p.functional("b", 0.4)
    assert np.all(c[p.slices["a"]] == 0) and c[p.slices["b"]].sum() == pytest.approx(1.0)
