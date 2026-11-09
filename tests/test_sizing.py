"""sized_mesh, truncation_radius, Mesh.element_of and Mesh.truncated."""

import numpy as np
import pytest

import fem_1d as fem

MANTLE = [0.546, 0.896, 1.0]  # CMB, 660, surface, in planet radii


# --- sized_mesh -----------------------------------------------------------


@pytest.mark.parametrize("drmax", [0.3, 0.1, 0.013])
def test_sized_mesh_respects_drmax_and_breakpoints(drmax):
    mesh = fem.sized_mesh(MANTLE, drmax=drmax)
    assert np.all(mesh.he <= drmax * (1 + 1e-12))
    for b in MANTLE:
        assert np.isclose(mesh.xe, b).any()
    assert mesh.xe[0] == MANTLE[0] and mesh.xe[-1] == MANTLE[-1]


def test_sized_mesh_is_uniform_within_each_span():
    mesh = fem.sized_mesh(MANTLE, drmax=0.05)
    lower = mesh.he[mesh.xe[:-1] < 0.896 - 1e-12]
    upper = mesh.he[mesh.xe[:-1] >= 0.896 - 1e-12]
    assert np.allclose(lower, lower[0]) and np.allclose(upper, upper[0])
    assert len(lower) == int(np.ceil(0.35 / 0.05)) and len(upper) == int(np.ceil(0.104 / 0.05))


def test_sized_mesh_coarse_drmax_gives_one_element_per_span():
    assert fem.sized_mesh(MANTLE, drmax=10.0).n_elements == 2


def test_sized_mesh_lmax_rule():
    mesh = fem.sized_mesh(MANTLE, lmax=19)
    assert np.all(mesh.he <= 0.1 * 1.0 / 20 * (1 + 1e-12))
    finer = fem.sized_mesh(MANTLE, lmax=39)
    assert finer.n_elements > mesh.n_elements
    looser = fem.sized_mesh(MANTLE, lmax=19, per_degree=0.4)
    assert looser.n_elements < mesh.n_elements


@pytest.mark.parametrize("kw", [{}, {"drmax": 0.1, "lmax": 4}, {"drmax": 0.0}, {"drmax": -1.0}, {"lmax": -1}])
def test_sized_mesh_bad_sizing_raises(kw):
    with pytest.raises(ValueError):
        fem.sized_mesh(MANTLE, **kw)


@pytest.mark.parametrize("b", [[1.0], [0.0, 1.0, 1.0], [1.0, 0.5]])
def test_sized_mesh_bad_breakpoints_raise(b):
    with pytest.raises(ValueError):
        fem.sized_mesh(b, drmax=0.1)


# --- truncation_radius ----------------------------------------------------


@pytest.mark.parametrize("l", [0, 2, 10, 100, 1000])
def test_truncation_radius_definition(l):
    r = fem.truncation_radius(l, 2.0, eps=1e-6)
    assert (r / 2.0) ** (l + 1) == pytest.approx(1e-6)


def test_truncation_radius_rises_with_degree():
    rs = [fem.truncation_radius(l, 1.0) for l in (1, 10, 100, 1000)]
    assert all(b > a for a, b in zip(rs, rs[1:])) and rs[-1] < 1.0


@pytest.mark.parametrize("l, eps", [(-1, 1e-8), (2, 0.0), (2, 1.0)])
def test_truncation_radius_bad_input(l, eps):
    with pytest.raises(ValueError):
        fem.truncation_radius(l, 1.0, eps=eps)


# --- element_of and truncated ---------------------------------------------


def test_element_of_interior_and_boundaries():
    mesh = fem.Mesh([0.0, 1.0, 2.0, 3.0])
    assert list(mesh.element_of([0.0, 0.5, 1.0, 2.5, 3.0])) == [0, 0, 1, 2, 2]
    assert list(mesh.element_of([0.0, 1.0, 2.0, 3.0], side="below")) == [0, 0, 1, 2]
    assert isinstance(mesh.element_of(1.5), int)


def test_element_of_snaps_points_within_rounding_of_a_boundary():
    mesh = fem.Mesh([0.0, 0.1, 0.3])
    assert mesh.element_of(0.1 - 1e-15) == 1 and mesh.element_of(0.1 + 1e-15, side="below") == 0


@pytest.mark.parametrize("x, side", [(-0.1, "above"), (3.5, "above"), (1.0, "left")])
def test_element_of_errors(x, side):
    with pytest.raises(ValueError):
        fem.Mesh([0.0, 1.0, 2.0, 3.0]).element_of(x, side)


def test_truncated_keeps_element_containing_point():
    mesh = fem.sized_mesh(MANTLE, drmax=0.05)
    r = fem.truncation_radius(50, 1.0)
    sub = mesh.truncated(r)
    assert sub.xe[0] <= r < sub.xe[1]
    assert np.allclose(sub.xe, mesh.xe[mesh.xe >= sub.xe[0] - 1e-15])


def test_truncated_below_mesh_is_identity_and_on_boundary_starts_there():
    mesh = fem.Mesh([0.0, 1.0, 2.0, 3.0])
    assert mesh.truncated(-5.0) is mesh
    assert np.array_equal(mesh.truncated(1.0).xe, [1.0, 2.0, 3.0])
    assert np.array_equal(mesh.truncated(3.0).xe, [2.0, 3.0])


def test_truncated_degree_l_problem_matches_full_problem():
    # -(r^2 u')' + l(l+1) u = 0 with u(a) = 1: solution r^l, negligible deep inside at high l
    l = 40
    full = fem.sized_mesh([0.1, 1.0], drmax=0.02)
    terms = [fem.Term(lambda r: r**2, 1, 1), fem.Term(lambda r: l * (l + 1.0))]

    def solve(mesh):
        s = fem.FunctionSpace.build(mesh, 6)
        p = fem.Problem(s, terms, [], [fem.Dirichlet("u", "left", 0.0), fem.Dirichlet("u", "right", 1.0)])
        return s, p.solve()

    s_full, sol_full = solve(full)
    s_cut, sol_cut = solve(full.truncated(fem.truncation_radius(l, 1.0, eps=1e-10)))
    x = np.linspace(s_cut.x[0], 1.0, 50)
    assert np.allclose(sol_cut.evaluate("u", x), sol_full.evaluate("u", x), atol=1e-9)
    assert np.allclose(sol_full.evaluate("u", x), x**l, atol=1e-6)
    assert s_cut.n_dofs < s_full.n_dofs / 2
