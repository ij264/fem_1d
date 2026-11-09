import numpy as np
import pytest

import fem_1d as fem

pi = np.pi
XE = np.linspace(0.0, 1.0, 5)  # element boundaries at 0, .25, .5, .75, 1


def test_continuous_space_shares_end_dofs():
    s = fem.FunctionSpace.build(XE, 3)
    n, Ne = 3, s.mesh.n_elements
    assert not s.discontinuous
    assert np.array_equal(s.conn, np.arange(Ne)[:, None] * n + np.arange(n + 1))
    assert s.n_dofs == Ne * n + 1


@pytest.mark.parametrize("n", [1, 2, 4])
def test_discontinuous_space_gives_each_element_its_own_dofs(n):
    s = fem.FunctionSpace.build(XE, n, discontinuous=True)
    assert s.n_dofs == 4 * (n + 1)
    assert np.array_equal(s.conn, np.arange(4 * (n + 1)).reshape(4, n + 1))
    assert not np.any(s.conn[:-1, -1] == s.conn[1:, 0])


def test_discontinuous_boundary_coordinates_repeated():
    s = fem.FunctionSpace.build(XE, 2, discontinuous=True)
    for x in XE[1:-1]:
        assert np.sum(np.isclose(s.x, x)) == 2
    assert np.all(np.diff(s.x) >= 0)


# --- dof_at ---------------------------------------------------------------


def test_dof_at_ends_and_nodes():
    s = fem.FunctionSpace.build(XE, 2)
    assert s.dof_at("left") == 0 and s.dof_at("right") == s.n_dofs - 1
    assert s.dof_at(0.0) == 0 and s.dof_at(1.0) == s.n_dofs - 1
    assert s.dof_at(0.125) == 1  # element midpoint node
    assert s.dof_at(0.5) == 4


def test_dof_at_ends_and_interior_nodes_of_discontinuous_space():
    s = fem.FunctionSpace.build(XE, 2, discontinuous=True)
    assert s.dof_at("left") == 0 and s.dof_at("right") == s.n_dofs - 1
    assert s.dof_at(0.125) == 1
    with pytest.raises(ValueError):
        s.dof_at(0.5)  # two dofs there


@pytest.mark.parametrize("at", [0.3, "middle", 2.0])
def test_dof_at_invalid(at):
    with pytest.raises(ValueError):
        fem.FunctionSpace.build(XE, 2).dof_at(at)


# --- assembly with broken spaces -----------------------------------------


def test_fully_broken_mass_matrix_is_block_diagonal():
    s = fem.FunctionSpace.build(XE, 2, discontinuous=True)
    one = lambda x: 1.0
    A, _ = fem.assemble_system(s, [fem.Term(one, 1, 1)])
    dense = A.toarray()
    for e in range(4):
        rows = s.conn[e]
        others = np.setdiff1d(np.arange(s.n_dofs), rows)
        assert np.abs(dense[np.ix_(rows, others)]).max() == 0


def test_l2_projection_of_step_is_exact_only_when_discontinuous():
    step = lambda x: np.where(x < 0.5, 1.0, 3.0)
    for discontinuous in (False, True):
        s = fem.FunctionSpace.build(XE, 2, discontinuous=discontinuous)
        M, _ = fem.assemble_system(s, [fem.Term(lambda x: 1.0)])
        b = fem.assemble_load(s, [fem.LoadTerm(step)])
        p = np.linalg.solve(M.toarray(), b)
        exact = discontinuous
        assert np.allclose(p, per_element(s, step), atol=1e-12) == exact


def per_element(s, f):
    """Expected nodal values of a piecewise function, using each element's own side."""
    out = np.empty(s.n_dofs)
    for e, dofs in enumerate(s.conn):
        mid = 0.5 * (s.mesh.xe[e] + s.mesh.xe[e + 1])
        out[dofs] = f(mid)
    return out


def mixed_poisson(n, Ne):
    # sigma = u', -sigma' = f with u = 0 imposed weakly; u is fully discontinuous
    xe = np.linspace(0, 1, Ne + 1)
    spaces = {
        "s": fem.FunctionSpace.build(xe, n),
        "u": fem.FunctionSpace.build(xe, n - 1, discontinuous=True),
    }
    terms = [
        fem.Term(lambda x: 1.0, 0, 0, test="s", trial="s"),
        fem.Term(lambda x: 1.0, 1, 0, test="s", trial="u"),
        fem.Term(lambda x: -1.0, 0, 1, test="u", trial="s"),
    ]
    f = lambda x: pi**2 * np.sin(pi * x)
    sol = fem.Problem(spaces, terms, [fem.LoadTerm(f, test="u")]).solve()
    return spaces, sol


def test_mixed_poisson_with_discontinuous_scalar_converges():
    errs = []
    for n in (3, 5, 7):
        spaces, sol = mixed_poisson(n, 3)
        errs.append(np.abs(sol["u"] - np.sin(pi * spaces["u"].x)).max())
    assert all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 1e-6


def test_mixed_poisson_flux_is_derivative():
    spaces, sol = mixed_poisson(6, 4)
    assert np.abs(sol["s"] - pi * np.cos(pi * spaces["s"].x)).max() < 1e-5


def test_pressure_like_field_can_jump_at_viscosity_interface():
    # Cartesian analogue of a pressure jump: the flux mu u' is continuous while
    # mu jumps, so u' jumps. Projecting u' onto a discontinuous field recovers it
    # exactly on both sides of the interface; a continuous field cannot.
    d, mu1, mu2 = 0.5, 1.0, 4.0
    mu = lambda x: np.where(x < d, mu1, mu2)
    s = fem.FunctionSpace.build(XE, 2)
    p = fem.Problem(s, [fem.Term(mu, 1, 1)], [], [fem.Dirichlet("u", "left", 0.0),
                                                  fem.Dirichlet("u", "right", 1.0)])
    u = p.solve()["u"]
    flux = 1.0 / (d / mu1 + (1 - d) / mu2)
    for discontinuous in (False, True):
        g = fem.FunctionSpace.build(XE, 1, discontinuous=discontinuous)
        spaces = {"u": s, "g": g}
        M, sl = fem.assemble_system(spaces, [fem.Term(lambda x: 1.0, test="g", trial="g"),
                                             fem.Term(lambda x: 1.0, 0, 1, test="g", trial="u")])
        gv = np.linalg.solve(M[sl["g"], sl["g"]].toarray(), M[sl["g"], sl["u"]] @ u)
        expected = per_element(g, lambda x: flux / mu(x))
        assert np.allclose(gv, expected, atol=1e-10) == discontinuous
