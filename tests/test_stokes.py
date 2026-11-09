"""Spherical Stokes: the adjoint assembled from its own weak form."""

import numpy as np
import pytest

import fem_1d as fem
from fem_1d import stokes

C, B, J_B, J_C = 0.55, 1.0, 1.3, -0.7
rho0 = lambda r: np.exp(0.4 * (1 - r))
kappa = lambda r: -0.4 * r
force = lambda r: np.sin(5 * r)
PARAMS = {"eta": lambda r: 1 + 10 * (r < 0.8)}


def spaces(n=5, Ne=6):
    mesh = fem.Mesh(np.linspace(C, B, Ne + 1))
    u = fem.FunctionSpace(mesh, fem.reference_element(n))
    p = fem.FunctionSpace(mesh, fem.reference_element(n - 2), discontinuous=True)
    return {"U": u, "V": u, "P": p}


def forward(params, l=2):
    terms = stokes.forward_terms(l, rho0, kappa)
    return fem.Problem(spaces(), terms, stokes.body_force(force), stokes.no_flow(), params)


def adjoint(params, l=2):
    terms = stokes.adjoint_terms(l, rho0, kappa)
    return fem.Problem(spaces(), terms, [], stokes.adjoint_constraints(J_B, J_C, B, C), params)


def J(params):
    T_b, T_c = stokes.tractions(forward(params).solve())
    return J_B * T_b + J_C * T_c


@pytest.mark.parametrize("l", [1, 2, 8])
def test_adjoint_weak_form_is_transpose_of_forward(l):
    fwd, adj = forward(PARAMS, l), adjoint(PARAMS, l)
    assert np.abs(adj.A - fwd.A.T).max() < 1e-10 * np.abs(fwd.A).max()


def test_incompressible_forward_is_symmetric():
    p = fem.Problem(spaces(), stokes.forward_terms(3), params=PARAMS)
    assert np.abs(p.A - p.A.T).max() < 1e-10


def test_adjoint_matches_discrete_adjoint():
    # T_b = reaction / b^2 and T_c = -reaction / c^2, so J has reaction weights r
    fwd, adj = forward(PARAMS), adjoint(PARAMS).solve()
    r = np.zeros(fwd.n_dofs)
    r[fwd.dof("U", "right")], r[fwd.dof("U", "left")] = J_B / B**2, -J_C / C**2
    lam = fwd.adjoint(fwd.A.T @ r) - r
    assert np.allclose(adj.u, lam, rtol=0, atol=1e-10 * np.abs(lam).max())


def test_reciprocity_J_equals_adjoint_dot_load():
    fwd = forward(PARAMS)
    assert adjoint(PARAMS).solve().u @ fwd.b == pytest.approx(J(PARAMS), rel=1e-10)


@pytest.mark.parametrize("relative", [False, True])
def test_viscosity_kernel_matches_finite_difference(relative):
    fwd = forward(PARAMS)
    K = fwd.kernel("eta", adjoint=adjoint(PARAMS).solve(), relative=relative)
    dm = lambda r: np.exp(-(((r - 0.7) / 0.05) ** 2))
    eta, eps = PARAMS["eta"], 1e-4  # smaller steps lose J to roundoff in the reactions
    step = (lambda e: lambda r: eta(r) * (1 + e * dm(r))) if relative else (lambda e: lambda r: eta(r) + e * dm(r))
    fd = (J({"eta": step(eps)}) - J({"eta": step(-eps)})) / (2 * eps)
    assert K.apply(dm) == pytest.approx(fd, rel=1e-6)


def test_kernel_rejects_adjoint_with_weights_or_wrong_size():
    fwd = forward(PARAMS)
    with pytest.raises(ValueError, match="not both"):
        fwd.kernel("eta", c=fwd.b, adjoint=np.zeros(fwd.n_dofs))
    with pytest.raises(ValueError, match="shape"):
        fwd.kernel("eta", adjoint=np.zeros(3))
