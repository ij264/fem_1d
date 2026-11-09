import numpy as np
import pytest
from scipy.linalg import LinAlgError

import fem_1d as fem

pi = np.pi
one = lambda x: 1.0


def space(n=3, Ne=4, **kw):
    return fem.FunctionSpace.build(np.linspace(0, 1, Ne + 1), n, **kw)


def both_ends(field="u", left=0.0, right=0.0):
    return [fem.Dirichlet(field, "left", left), fem.Dirichlet(field, "right", right)]


# --- constraint bookkeeping ----------------------------------------------


def test_dirichlet_both_ends():
    s = space()
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [], both_ends())
    assert list(p.fixed) == [0, s.n_dofs - 1]
    assert p.free.sum() == s.n_dofs - 2 and not p.free[0] and not p.free[-1]


def test_dirichlet_resolves_into_the_right_field_block():
    spaces = {"U": space(4), "P": space(2)}
    terms = [fem.Term(one, 1, 1, f, f) for f in spaces]
    p = fem.Problem(spaces, terms, [], [fem.Dirichlet("P", "right", 2.0), fem.Dirichlet("U", 0.5)])
    sl = p.slices
    assert set(p.fixed) == {sl["P"].stop - 1, sl["U"].start + spaces["U"].dof_at(0.5)}
    assert p.dof("P", "left") == sl["P"].start


def test_duplicate_equal_dirichlet_is_fine_but_conflict_raises():
    s = space()
    fem.Problem(s, [], [], [fem.Dirichlet("u", "left", 1.0), fem.Dirichlet("u", 0.0, 1.0)])
    with pytest.raises(ValueError):
        fem.Problem(s, [], [], [fem.Dirichlet("u", "left", 1.0), fem.Dirichlet("u", 0.0, 2.0)])


def test_dirichlet_at_element_boundary_of_discontinuous_space_raises():
    s = space(2, discontinuous=True)
    with pytest.raises(ValueError):
        fem.Problem(s, [], [], [fem.Dirichlet("u", 0.5)])
    p = fem.Problem(s, [], [], [fem.Dirichlet("u", "right")])
    assert list(p.fixed) == [s.n_dofs - 1]


def test_callable_dirichlet_and_point_values_read_params():
    s = space(1, 2)
    params = {"g": 3.0, "top": 2.0}
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.PointLoad(lambda prm: prm["g"], "right")],
                    [fem.Dirichlet("u", "left", lambda prm: prm["top"])], params=params)
    assert np.allclose(p.solve()["u"], 2.0 + 3.0 * s.x)


def test_unknown_constraint_type_raises():
    with pytest.raises(TypeError):
        fem.Problem(space(), [], [], ["u(0) = 1"])


def test_unknown_field_in_dirichlet_raises():
    with pytest.raises(KeyError):
        fem.Problem(space(), [], [], [fem.Dirichlet("nope")])


# --- forward solves -------------------------------------------------------


def test_nonzero_dirichlet_exact():
    s = space(2, 3)
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.LoadTerm(lambda x: -2.0)], both_ends(left=1.0, right=2.0))
    u = p.solve()["u"]
    assert np.allclose(u, s.x**2 + 1, atol=1e-12)


def test_solution_fields_and_getitem():
    spaces = {"a": space(2), "b": space(3)}
    terms = [fem.Term(one, 1, 1, f, f) for f in spaces]
    loads = [fem.LoadTerm(one, test=f) for f in spaces]
    sol = fem.Problem(spaces, terms, loads, both_ends("a") + both_ends("b")).solve()
    assert set(sol.fields) == {"a", "b"}
    assert np.array_equal(sol["b"], sol.u[sol.problem.slices["b"]])


def test_neumann_point_load():
    # -u'' = -2, u(0) = 0, u'(1) = 2  ->  u = x^2
    s = space(2, 3)
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.LoadTerm(lambda x: -2.0), fem.PointLoad(2.0, "right")],
                    [fem.Dirichlet("u", "left")])
    assert np.allclose(p.solve()["u"], s.x**2, atol=1e-12)


@pytest.mark.parametrize("alpha, g", [(0.0, 1.0), (2.0, 3.0), (-0.5, 1.0)])
def test_robin_condition_with_point_term(alpha, g):
    # -u'' = 0, u(0) = 0, u'(1) + alpha u(1) = g  ->  u = g x / (1 + alpha)
    s = space(1, 3)
    p = fem.Problem(s, [fem.Term(one, 1, 1), fem.PointTerm(alpha, "right")], [fem.PointLoad(g, "right")],
                    [fem.Dirichlet("u", "left")])
    assert np.allclose(p.solve()["u"], g * s.x / (1 + alpha), atol=1e-12)


def test_interior_point_load_gives_greens_function():
    # -u'' = delta(x - 0.25), u(0) = u(1) = 0
    s = space(1, 4)
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.PointLoad(1.0, 0.25)], both_ends())
    G = np.where(s.x < 0.25, s.x * 0.75, 0.25 * (1 - s.x))
    assert np.allclose(p.solve()["u"], G, atol=1e-12)


def test_point_term_couples_two_fields():
    # a(u, v) adds k * u(1) * w(1) to field w's equation
    spaces = {"u": space(2), "w": space(2)}
    terms = [fem.Term(one, 1, 1, f, f) for f in spaces] + [fem.PointTerm(3.0, "right", test="w", trial="u")]
    A, sl = fem.assemble_system(spaces, terms)
    i, j = sl["w"].stop - 1, sl["u"].stop - 1
    assert A[i, j] == 3.0 and A[j, i] == 0.0


def test_solve_with_other_load_reuses_factors():
    s = space()
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.LoadTerm(one)], both_ends())
    p.solve()
    F = p._factors()
    b2 = fem.assemble_load(s, [fem.LoadTerm(lambda x: x)])
    u2 = p.solve(b2)["u"]
    assert p._factors() is F
    assert np.allclose(u2, (s.x - s.x**3) / 6, atol=1e-6)


# --- reactions ------------------------------------------------------------


def test_reactions_are_outward_fluxes():
    # -(mu u')' = f with mu = 1 + x, u = x^2 + x: outward flux -mu u'(0) = -1, mu u'(1) = 6
    s = space(3, 3)
    mu = lambda x: 1 + x
    f = lambda x: -(4 * x + 3)
    p = fem.Problem(s, [fem.Term(mu, 1, 1)], [fem.LoadTerm(f)], both_ends(left=0.0, right=2.0))
    sol = p.solve()
    assert np.allclose(sol["u"], s.x**2 + s.x, atol=1e-12)
    assert sol.reaction("u", "left") == pytest.approx(-1.0)
    assert sol.reaction("u", "right") == pytest.approx(6.0)


def test_residual_vanishes_on_free_dofs():
    s = space()
    p = fem.Problem(s, [fem.Term(lambda x: 1 + x, 1, 1), fem.Term(lambda x: 2.0, 0, 1)],
                    [fem.LoadTerm(np.cos)], both_ends(right=1.0))
    sol = p.solve()
    assert np.abs(sol.residual()[p.free]).max() < 1e-12


def test_reaction_at_interior_dirichlet_is_flux_jump():
    # -u'' = 0 with u(0) = 0, u(0.5) = 1, u(1) = 0: hat function with slopes +2, -2
    s = space(1, 4)
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [], both_ends() + [fem.Dirichlet("u", 0.5, 1.0)])
    sol = p.solve()
    assert sol.reaction("u", 0.5) == pytest.approx(4.0)  # u'(0.5-) - u'(0.5+)


# --- singular systems ---------------------------------------------------


def test_pure_neumann_problem_is_singular():
    # -u'' = f with no Dirichlet condition: unique only up to a constant
    s = space(6, 4)
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.LoadTerm(lambda x: np.cos(pi * x))])
    K = p.A.toarray()
    assert np.linalg.matrix_rank(K) == K.shape[0] - 1


def test_exactly_singular_matrix_raises():
    s = space(2, 2)
    p = fem.Problem(s, [fem.Term(lambda x: 0.0)], [], both_ends())
    with pytest.raises(LinAlgError):
        p.solve()


# --- adjoint --------------------------------------------------------------


def nonsymmetric_problem(constraints):
    s = space(4, 5)
    terms = [fem.Term(lambda x: 1 + x, 1, 1), fem.Term(lambda x: 3.0, 0, 1), fem.Term(lambda x: 2.0)]
    return s, fem.Problem(s, terms, [fem.LoadTerm(np.exp)], constraints)


def test_adjoint_identity_J_equals_lam_dot_b():
    _, p = nonsymmetric_problem(both_ends())
    c = np.random.default_rng(0).standard_normal(p.n_dofs)
    lam = p.adjoint(c)
    assert lam @ p.b == pytest.approx(c @ p.solve().u)
    assert np.all(lam[~p.free] == 0)


def test_adjoint_matches_transpose_solve():
    _, p = nonsymmetric_problem(both_ends())
    c = np.random.default_rng(1).standard_normal(p.n_dofs)
    f = p.free
    expected = np.linalg.solve(p.A.toarray()[np.ix_(f, f)].T, c[f])
    assert np.allclose(p.adjoint(c)[f], expected)


def test_adjoint_many_functionals_at_once():
    _, p = nonsymmetric_problem(both_ends())
    C = np.random.default_rng(2).standard_normal((p.n_dofs, 5))
    L = p.adjoint(C)
    assert L.shape == C.shape
    assert np.allclose(L[:, 3], p.adjoint(C[:, 3]))


def test_adjoint_gives_response_to_load_at_every_dof():
    # one adjoint solve replaces one forward solve per load position
    s, p = nonsymmetric_problem(both_ends())
    c = np.zeros(p.n_dofs)
    c[p.dof("u", 0.6)] = 1.0
    lam = p.adjoint(c)
    for j in (3, 7, 12):
        b = np.zeros(p.n_dofs)
        b[j] = 1.0
        assert p.solve(b).u[p.dof("u", 0.6)] == pytest.approx(lam[j])
