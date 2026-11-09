"""Self-gravitating spherical Stokes (fem_1d.self_gravity). Written by Claude for Issac Jacob, Nov 2026."""

import numpy as np
import pytest
from scipy.integrate import quad

import fem_1d as fem
from fem_1d import self_gravity as sg
from fem_1d import stokes

C, B, L_DEG, R_670 = 0.55, 1.0, 2, 0.8
eta = lambda r: np.where(r < R_670, 30.0, 1.0)
rho0 = lambda r: np.exp(0.4 * (1 - r))
kappa = lambda r: -0.4 * r
drho = lambda r: np.exp(-(((r - 0.75) / 0.05) ** 2))
G = 1.0
g0 = sg.gravity_profile(rho0, C, B, G, rho_core=2 * rho0(C))
PARAMS = {"eta": eta, "drho": drho}
FLUIDS = dict(rho_fluid_b=0.3, rho_fluid_c=2.0)


def make(n=5, drmax=0.02):
    mesh = fem.sized_mesh([C, R_670, B], drmax=drmax)
    vel = fem.FunctionSpace(mesh, fem.reference_element(n))
    pres = fem.FunctionSpace(mesh, fem.reference_element(n - 2), discontinuous=True)
    return vel, pres, sg.spaces(vel, pres)


VEL, PRES, SPACES = make()


def fwd(params=PARAMS, l=L_DEG, **kw):
    return sg.forward_problem(SPACES, l, rho0, kappa, g0, G, B, C, params, **kw)


def adj(params=PARAMS, l=L_DEG, kw=None, **w):
    return sg.adjoint_problem(SPACES, l, rho0, kappa, g0, G, B, C, params, **w, **(kw or {}))


def observable(sol, j_b=0.0, j_c=0.0, j_phi=0.0):
    T_b, T_c = sg.tractions(sol)
    return j_b * T_b + j_c * T_c + j_phi * float(sg.potential(sol, B)[0])


def test_degree_one_is_rejected():
    with pytest.raises(ValueError, match="null mode"):
        sg.forward_terms(1, rho0, kappa, g0, G, B, C)


def test_G_zero_limit_matches_nongravitating_stokes():
    """With G = 0 and constant g0 the potential decouples and the tractions equal the reactions of fem_1d.stokes."""
    one = lambda r: 1.0 + 0.0 * r
    sol = sg.forward_problem(SPACES, L_DEG, rho0, kappa, one, 0.0, B, C, PARAMS).solve()
    ref = fem.Problem(
        {"U": VEL, "V": VEL, "P": PRES},
        stokes.forward_terms(L_DEG, rho0, kappa),
        stokes.body_force(lambda r: -one(r), param="drho"),
        stokes.no_flow(),
        PARAMS,
    ).solve()
    assert sg.tractions(sol) == pytest.approx(stokes.tractions(ref), rel=1e-8)
    assert np.abs(sol["Phi"]).max() == 0.0


def test_poisson_row_matches_analytic_shell_green_function():
    """Without flow: Phi_l(r) = -4 pi G/(2l+1) int r'^2 drho r_<^l / r_>^(l+1) dr'."""
    terms = [t for t in sg.forward_terms(L_DEG, rho0, kappa, g0, G, B, C) if t.test == "Phi" and t.trial == "Phi"]
    load = [t for t in sg.body_force(g0, G) if t.test == "Phi"]
    sol = fem.Problem({"Phi": VEL}, terms, load, (), PARAMS).solve()
    l = L_DEG
    for r in (0.6, 0.7, 0.75, 0.9, 1.0):
        f = lambda s: s**2 * drho(s) * (min(r, s) ** l / max(r, s) ** (l + 1))
        exact = -4 * np.pi * G / (2 * l + 1) * (quad(f, C, r, points=[0.75], limit=200)[0] + quad(f, r, B, points=[0.75], limit=200)[0])
        assert float(sol.evaluate("Phi", r)[0]) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("kw", [{}, FLUIDS])
@pytest.mark.parametrize("l", [2, 8])
def test_adjoint_weak_form_is_transpose_of_forward(kw, l):
    f, a = fwd(l=l, **kw), adj(l=l, kw=kw, j_b=1.0)
    assert np.abs(a.A - f.A.T).max() < 1e-12 * np.abs(f.A).max()


@pytest.mark.parametrize("kw", [{}, FLUIDS])
@pytest.mark.parametrize("w", [dict(j_b=1.0), dict(j_c=1.0), dict(j_phi=1.0), dict(j_b=0.4, j_c=-1.3, j_phi=2.0)])
def test_reciprocity_J_equals_adjoint_dot_load(w, kw):
    f = fwd(**kw)
    lam = adj(kw=kw, **w).solve()
    assert lam.u @ f.b == pytest.approx(observable(f.solve(), **w), rel=1e-8, abs=1e-14)  # cond(A) ~ 1e10


@pytest.mark.parametrize("w", [dict(j_b=1.0), dict(j_phi=1.0)])
def test_viscosity_kernel_matches_finite_difference(w):
    K = fwd(**FLUIDS).kernel("eta", adjoint=adj(kw=FLUIDS, **w).solve(), relative=True)
    dm = lambda r: np.exp(-(((r - 0.65) / 0.05) ** 2))
    eps = 1e-4
    J = lambda s: observable(fwd({"eta": lambda r: eta(r) * (1 + s * dm(r)), "drho": drho}, **FLUIDS).solve(), **w)
    assert K.apply(dm) == pytest.approx((J(eps) - J(-eps)) / (2 * eps), rel=1e-6)
    assert K.apply(1.0) == pytest.approx(0.0, abs=1e-9)  # the overall viscosity scale drops out


@pytest.mark.parametrize("w", [dict(j_b=1.0), dict(j_phi=1.0)])
def test_density_kernel_matches_finite_difference(w):
    K = fwd(**FLUIDS).kernel("drho", adjoint=adj(kw=FLUIDS, **w).solve())
    dm = lambda r: np.exp(-(((r - 0.7) / 0.05) ** 2))
    eps = 1e-4
    J = lambda s: observable(fwd({"eta": eta, "drho": lambda r: drho(r) + s * dm(r)}, **FLUIDS).solve(), **w)
    assert K.apply(dm) == pytest.approx((J(eps) - J(-eps)) / (2 * eps), rel=1e-6)


def test_density_kernel_is_the_adjoint_field():
    """K_drho = -r^2 (g0 U+ + 4 pi G Phi+) at the quadrature points."""
    lam = adj(j_phi=1.0).solve()
    K = fwd().kernel("drho", adjoint=lam)
    keep = np.ones_like(K.x, dtype=bool)  # interior points: at element ends evaluate() picks one side of a node
    keep[:, [0, -1]] = False
    x, Kv = K.x[keep], K.values[keep]
    U, Phi = lam.evaluate("U", x), lam.evaluate("Phi", x)
    assert np.allclose(Kv, -(x**2) * (g0(x) * U + 4 * np.pi * G * Phi), rtol=1e-8, atol=1e-12 * np.abs(Kv).max())


def test_mass_sheets_compensate_the_direct_potential():
    """Without feedback the flow is unchanged; a dense anomaly depresses both boundaries, and their negative
    mass sheets reduce the magnitude of the (negative) direct potential at the surface."""
    with_sheets, without = fwd(feedback=0.0).solve(), fwd(feedback=0.0, sheets=0.0).solve()
    T_b, T_c = sg.tractions(with_sheets)
    assert T_b > 0 and T_c < 0
    assert sg.tractions(with_sheets) == pytest.approx(sg.tractions(without), rel=1e-10)
    direct, total = float(sg.potential(without, B)[0]), float(sg.potential(with_sheets, B)[0])
    assert direct < 0 and direct < total and abs(total) < abs(direct)


def test_traction_converges_under_refinement():
    out = []
    for n, drmax in [(5, 0.02), (6, 0.01), (7, 0.01)]:
        _, _, sp = make(n, drmax)
        out.append(sg.tractions(sg.forward_problem(sp, L_DEG, rho0, kappa, g0, G, B, C, PARAMS).solve())[0])
    assert abs(out[1] - out[2]) < 1e-5 * abs(out[2])
    assert abs(out[0] - out[2]) < 1e-3 * abs(out[2])


def test_geoid_weight_is_bruns():
    sol = fwd().solve()
    w = sg.geoid_weight(g0, B)
    assert observable(sol, j_phi=w) == pytest.approx(-float(sg.potential(sol, B)[0]) / float(g0(B)))
