"""Self-gravitating instantaneous Stokes flow in a spherical shell c < r < b, at one
spherical-harmonic degree l, and its adjoint for traction and potential observables.

Written by Claude (AI assistant) for Issac Jacob, Nov 2026, extending
:mod:`fem_1d.stokes`; checked against the propagator/BVP kernels of Sia's code.

Fields are U, V, P as in :mod:`fem_1d.stokes`, the perturbation of the
gravitational potential Phi (physics sign: laplacian Phi = 4 pi G rho), and the
two boundary tractions T_b, T_c, held in a two-dof field "T". The tractions
are unknowns of the system, not reactions, because the mass of the deflected
boundaries enters Poisson's equation through them.

With L = l (l + 1), the forward weak form is: find (U, V, P, Phi, T) such that,
for all test fields (U~, V~, P~, Phi~, T~),

    int 2 eta r^2 B(u, u~) dr - int r^2 P div u~ dr
      + int r^2 rho0 [Phi' U~ + (L / r) Phi V~] dr
      - b^2 T_b U~(b) + c^2 T_c U~(c)                    = -int r^2 g0 drho U~ dr

    - int r^2 P~ rho0 [div u + kappa U / r] dr           = 0

    - b^2 T~_b U(b) + c^2 T~_c U(c)                      = 0

    int (r^2 Phi' Phi~' + L Phi Phi~) dr + (l + 1) b Phi(b) Phi~(b) + l c Phi(c) Phi~(c)
      + 4 pi G [b^2 sigma_b Phi~(b) + c^2 sigma_c Phi~(c)] = -4 pi G int r^2 drho Phi~ dr,

with B the strain pairing of :mod:`fem_1d.stokes`. The last two boundary terms of
the Poisson row match Phi to the potentials outside (decaying as r^-(l+1)) and in
the core (regular, r^l). The surface mass densities of the deflected boundaries are

    sigma_b = -(T_b - rho_fluid_b Phi(b)) / g0(b),   sigma_c = +(T_c - rho_fluid_c Phi(c)) / g0(c),

where rho_fluid_c is the density of the fluid core and rho_fluid_b that of a
hydrostatic fluid above the surface whose own redistribution is not counted as
mass. An ocean that follows the geoid needs rho_fluid_b = 0 (the default): its
change of thickness adds rho_w N to the surface mass, which cancels the rho_w
term, so sigma_b = -T_b / g0(b). This choice matches the kernels of Sia's code; setting
rho_fluid_b to the water density does not (geoid off by ~18% at l = 2). Both are
0 for void boundaries. Each density contrast across a boundary cancels between
the topography and its mass sheet.

The observable is J = j_b T_b + j_c T_c + j_phi Phi(b). The adjoint weak form,
assembled by :func:`adjoint_terms`, is the forward bilinear form with the roles
of the unknowns and the test fields exchanged, and the observable as its load
on the T and Phi rows (:func:`observable_loads`). Then

    dJ = int K_drho d(drho) dr,    K_drho = -r^2 (g0 U+ + 4 pi G Phi+),
    dJ = int K_eta d(ln eta) dr,   K_eta = -2 eta r^2 B(u, u+),

are ``forward.kernel("drho", adjoint=...)`` and ``forward.kernel("eta", adjoint=..., relative=True)``.

Degree 1 is rejected: the self-gravitating shell then admits a rigid
translation that these boundary conditions do not remove.
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from . import stokes
from ._typing import Coefficient
from .assembly import LoadTerm, PointLoad, PointTerm, Term
from .problem import Problem, Solution
from .space import FunctionSpace


class TractionSpace:
    """The two boundary tractions as a field: T_b at "right" (r = b), T_c at "left" (r = c).

    It has no basis functions, only two dofs reached by point terms, so it can
    be stacked with the other fields of a :class:`~fem_1d.Problem`.

    Parameters
    ----------
    like : FunctionSpace
        Any field on the shell; only its mesh is used.

    Examples
    --------
    >>> import numpy as np
    >>> from fem_1d import FunctionSpace
    >>> T = TractionSpace(FunctionSpace.build(np.linspace(0.5, 1.0, 3), 2))
    >>> T.n_dofs, T.dof_at("right"), T.dof_at("left")
    (2, 0, 1)
    """

    n_dofs = 2

    def __init__(self, like: FunctionSpace):
        self.mesh, self.ref = like.mesh, like.ref

    def dof_at(self, at: str):
        """Dof index of the traction at ``at`` ("right" for T_b, "left" for T_c)."""
        if at not in ("right", "left"):
            raise ValueError(f"tractions live at 'right' or 'left', got {at!r}")
        return {"right": 0, "left": 1}[at]


def spaces(velocity: FunctionSpace, pressure: FunctionSpace):
    """Fields of the self-gravitating problem: U, V, Phi on ``velocity``, P on ``pressure``, and T.

    Examples
    --------
    >>> import numpy as np
    >>> from fem_1d import FunctionSpace
    >>> v = FunctionSpace.build(np.linspace(0.5, 1.0, 3), 3)
    >>> p = FunctionSpace.build(np.linspace(0.5, 1.0, 3), 1, discontinuous=True)
    >>> list(spaces(v, p))
    ['U', 'V', 'P', 'Phi', 'T']
    """
    return {"U": velocity, "V": velocity, "P": pressure, "Phi": velocity, "T": TractionSpace(velocity)}


def gravity_profile(rho0: Callable, c: float, b: float, G: float, rho_core: float, n: int = 20001):
    """Reference gravity g0(r) = G M(r) / r^2 of the shell density rho0 over a uniform core.

    Parameters
    ----------
    rho0 : callable
        Reference density of the shell.
    c, b : float
        Inner and outer radius.
    G : float
        Gravitational constant.
    rho_core : float
        Mean density of the core.
    n : int, default 20001
        Points of the trapezoidal mass integration.

    Returns
    -------
    scipy.interpolate.CubicSpline
        g0 on [c, b].

    Examples
    --------
    A uniform planet of density 1 has g0 = (4 pi / 3) G r:

    >>> import numpy as np
    >>> g = gravity_profile(lambda r: 1.0 + 0 * r, 0.5, 1.0, 1.0, rho_core=1.0)
    >>> bool(np.isclose(g(0.8), 4 * np.pi / 3 * 0.8))
    True
    """
    r = np.linspace(c, b, n)
    m = (4 * np.pi / 3) * rho_core * c**3 + cumulative_trapezoid(4 * np.pi * rho0(r) * r**2, r, initial=0.0)
    return CubicSpline(r, G * m / r**2)


def _check_degree(l: int):
    if l < 2:
        raise ValueError(f"degree {l} < 2: the self-gravitating shell has a rigid-translation null mode at l = 1")


def _as_fn(f: Coefficient | float):
    return f if callable(f) else (lambda r, v=float(f): v + 0.0 * r)


def _gravity_terms(l, rho0, g0, G, b, c, rho_fluid_b, rho_fluid_c, feedback, sheets):
    """Every forward term involving Phi or T, as (coeff or value, kind, test, d_test, trial, d_trial, at)."""
    L = l * (l + 1)
    gb, gc, k = float(g0(b)), float(g0(c)), 4 * np.pi * G
    out = [
        # gravity acting on the flow: int r^2 rho0 [Phi' U~ + (L/r) Phi V~] dr
        (lambda r: feedback * r**2 * rho0(r), "vol", "U", 0, "Phi", 1, None),
        (lambda r: feedback * L * r * rho0(r), "vol", "V", 0, "Phi", 0, None),
        # Poisson: int (r^2 Phi' Phi~' + L Phi Phi~) dr and the matching to the outside potentials
        (lambda r: r**2, "vol", "Phi", 1, "Phi", 1, None),
        (lambda r: L + 0.0 * r, "vol", "Phi", 0, "Phi", 0, None),
        ((l + 1) * b, "pt", "Phi", 0, "Phi", 0, "right"),
        (l * c, "pt", "Phi", 0, "Phi", 0, "left"),
        # tractions in the momentum rows, and the no-flow rows tested by T~
        (-(b**2), "pt", "U", 0, "T", 0, "right"),
        (+(c**2), "pt", "U", 0, "T", 0, "left"),
        (-(b**2), "pt", "T", 0, "U", 0, "right"),
        (+(c**2), "pt", "T", 0, "U", 0, "left"),
        # mass sheets in the Poisson rows: 4 pi G b^2 sigma_b Phi~(b) + 4 pi G c^2 sigma_c Phi~(c)
        (-sheets * k * b**2 / gb, "pt", "Phi", 0, "T", 0, "right"),
        (+sheets * k * c**2 / gc, "pt", "Phi", 0, "T", 0, "left"),
    ]
    if rho_fluid_b:
        out.append((+sheets * k * b**2 * rho_fluid_b / gb, "pt", "Phi", 0, "Phi", 0, "right"))
    if rho_fluid_c:
        out.append((-sheets * k * c**2 * rho_fluid_c / gc, "pt", "Phi", 0, "Phi", 0, "left"))
    return out


def _build(entries, swap: bool):
    """Terms from the entries; with swap, test and trial are exchanged (the adjoint weak form)."""
    terms = []
    for v, kind, te, dte, tr, dtr, at in entries:
        if swap:
            te, dte, tr, dtr = tr, dtr, te, dte
        if kind == "vol":
            terms.append(Term(v, dte, dtr, test=te, trial=tr))
        else:
            terms.append(PointTerm(v, at, test=te, trial=tr))
    return terms


def forward_terms(
    l: int,
    rho0: Coefficient | float,
    kappa: Coefficient | float,
    g0: Callable,
    G: float,
    b: float,
    c: float,
    rho_fluid_b: float = 0.0,
    rho_fluid_c: float = 0.0,
    feedback: float = 1.0,
    sheets: float = 1.0,
):
    """Bilinear form of the self-gravitating forward problem, viscosity tagged "eta".

    Parameters
    ----------
    l : int
        Spherical-harmonic degree (>= 2).
    rho0, kappa : callable or float
        Reference density and anelastic coefficient, as in :func:`fem_1d.stokes.forward_terms`.
    g0 : callable
        Reference gravity, e.g. from :func:`gravity_profile`.
    G : float
        Gravitational constant.
    b, c : float
        Outer and inner radius.
    rho_fluid_b, rho_fluid_c : float, default 0.0
        Density of a hydrostatic fluid above the surface whose redistribution is
        not counted as mass, and of the fluid core. Keep rho_fluid_b = 0 for an
        ocean that follows the geoid (see the module docstring); 0 for a void.
    feedback, sheets : float, default 1.0
        Scale the two gravity couplings (potential acting on the flow; boundary
        masses in Poisson's equation). Diagnostics only: 1 is the physical problem.

    Returns
    -------
    list of Term and PointTerm
        On fields "U", "V", "P", "Phi" and "T" (see :func:`spaces`).

    Raises
    ------
    ValueError
        If l < 2.

    Examples
    --------
    >>> import numpy as np
    >>> g = gravity_profile(lambda r: 1.0 + 0 * r, 0.5, 1.0, 1.0, rho_core=2.0)
    >>> terms = forward_terms(2, 1.0, 0.0, g, 1.0, 1.0, 0.5)
    >>> sorted({t.test for t in terms})
    ['P', 'Phi', 'T', 'U', 'V']
    """
    _check_degree(l)
    rho0, kappa = _as_fn(rho0), _as_fn(kappa)
    entries = _gravity_terms(l, rho0, g0, G, b, c, rho_fluid_b, rho_fluid_c, feedback, sheets)
    return list(stokes.forward_terms(l, rho0, kappa)) + _build(entries, swap=False)


def adjoint_terms(
    l: int,
    rho0: Coefficient | float,
    kappa: Coefficient | float,
    g0: Callable,
    G: float,
    b: float,
    c: float,
    rho_fluid_b: float = 0.0,
    rho_fluid_c: float = 0.0,
    feedback: float = 1.0,
    sheets: float = 1.0,
):
    """Bilinear form of the adjoint problem, viscosity tagged "eta".

    The adjoint unknowns (u+, P+, Phi+, lam) are the trial fields and the
    arbitrary fields the test fields, under the same names. The flow part is
    :func:`fem_1d.stokes.adjoint_terms`; every gravity and traction term has its
    test and trial roles exchanged. Parameters as for :func:`forward_terms`.

    Examples
    --------
    Assembled, it is the transpose of the forward matrix:

    >>> import numpy as np
    >>> from fem_1d import FunctionSpace, assemble_system
    >>> v = FunctionSpace.build(np.linspace(0.5, 1.0, 3), 3)
    >>> p = FunctionSpace.build(np.linspace(0.5, 1.0, 3), 1, discontinuous=True)
    >>> g = gravity_profile(lambda r: 1.0 + 0 * r, 0.5, 1.0, 1.0, rho_core=2.0)
    >>> args = (2, 1.0, -0.2, g, 1.0, 1.0, 0.5)
    >>> A, _ = assemble_system(spaces(v, p), forward_terms(*args), {"eta": 1.0})
    >>> A_adj, _ = assemble_system(spaces(v, p), adjoint_terms(*args), {"eta": 1.0})
    >>> bool(np.allclose(A_adj.toarray(), A.T.toarray()))
    True
    """
    _check_degree(l)
    rho0, kappa = _as_fn(rho0), _as_fn(kappa)
    entries = _gravity_terms(l, rho0, g0, G, b, c, rho_fluid_b, rho_fluid_c, feedback, sheets)
    return list(stokes.adjoint_terms(l, rho0, kappa)) + _build(entries, swap=True)


def body_force(g0: Callable, G: float, param: str | None = "drho"):
    """Forward loads of a density anomaly drho: buoyancy -int r^2 g0 drho U~ dr and
    its potential -4 pi G int r^2 drho Phi~ dr, both tagged ``param``.

    Examples
    --------
    >>> [t.test for t in body_force(lambda r: 1.0, 1.0)]
    ['U', 'Phi']
    """
    return [
        LoadTerm(lambda r: -(r**2) * g0(r), test="U", param=param),
        LoadTerm(lambda r: -4 * np.pi * G * r**2 + 0.0 * r, test="Phi", param=param),
    ]


def observable_loads(j_b: float = 0.0, j_c: float = 0.0, j_phi: float = 0.0):
    """Adjoint loads for J = j_b T_b + j_c T_c + j_phi Phi(b), on the T and Phi rows.

    Examples
    --------
    >>> [(t.test, t.at) for t in observable_loads(1.0)]
    [('T', 'right'), ('T', 'left'), ('Phi', 'right')]
    """
    return [PointLoad(j_b, "right", test="T"), PointLoad(j_c, "left", test="T"), PointLoad(j_phi, "right", test="Phi")]


def forward_problem(fields, l, rho0, kappa, g0, G, b, c, params, **kw):
    """The forward :class:`~fem_1d.Problem` (no Dirichlet conditions: the tractions hold U(b) = U(c) = 0).

    ``fields`` comes from :func:`spaces`; ``params`` must give "eta" and "drho";
    keyword arguments go to :func:`forward_terms`.
    """
    return Problem(fields, forward_terms(l, rho0, kappa, g0, G, b, c, **kw), body_force(g0, G), (), params)


def adjoint_problem(fields, l, rho0, kappa, g0, G, b, c, params, j_b=0.0, j_c=0.0, j_phi=0.0, **kw):
    """The adjoint :class:`~fem_1d.Problem` for J = j_b T_b + j_c T_c + j_phi Phi(b)."""
    return Problem(fields, adjoint_terms(l, rho0, kappa, g0, G, b, c, **kw), observable_loads(j_b, j_c, j_phi), (), params)


def tractions(solution: Solution):
    """(T_b, T_c) of a forward solution, or (lam_b, lam_c) of an adjoint one."""
    T = solution["T"]
    return float(T[0]), float(T[1])


def potential(solution: Solution, r):
    """Phi at radius r (float or array)."""
    return solution.evaluate("Phi", r)


def geoid_weight(g0: Callable, b: float):
    """j_phi for the geoid height N = -Phi(b) / g0(b) (Bruns' formula, physics-sign potential)."""
    return -1.0 / float(g0(b))
