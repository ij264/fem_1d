"""Instantaneous (anelastic) Stokes flow in a spherical shell c < r < b, reduced
to one spherical-harmonic degree l, and its adjoint for traction observables.

Fields are the radial and tangential velocity u = (U, V), the pressure P, and
the radial tractions T_b, T_c on the outer and inner boundaries, which hold
U(b) = U(c) = 0. With L = l (l + 1), the strain pairing of two flows is::

    B(a, b) = U_a' U_b' + F_a F_b / 2 + L X_a X_b / 2 + L (L - 2) V_a V_b / (2 r^2)
              - (div a)(div b) / 3,

    F = (2 U - L V) / r,   X = V' - V / r + U / r,   div a = U_a' + F_a.

Forward weak form, for all test fields (u~, P~, T~_b, T~_c)::

    int 2 eta r^2 B(u, u~) dr - int r^2 P div u~ dr
      - int r^2 P~ rho0 [div u + kappa U / r] dr
      - b^2 T_b U~(b) + c^2 T_c U~(c) - b^2 T~_b U(b) + c^2 T~_c U(c)
    = int r^2 f U~ dr,

with f the radial body force (e.g. -g drho). The observable is
J = j_b T_b + j_c T_c, and the adjoint weak form is: find
(u+, P+, lam_b, lam_c) such that, for all (u~, P~, T~_b, T~_c)::

    int 2 eta r^2 B(u~, u+) dr - int r^2 P~ div u+ dr
      - int r^2 P+ rho0 [div u~ + kappa U~ / r] dr
      - b^2 T~_b U+(b) + c^2 T~_c U+(c) - b^2 lam_b U~(b) + c^2 lam_c U~(c)
    = j_b T~_b + j_c T~_c.

Both are assembled from these weak forms directly. The tractions are not
fields: the rows tested by T~ say U(b) = U(c) = 0 in the forward problem and
-b^2 U+(b) = j_b, c^2 U+(c) = j_c in the adjoint, which are applied as
Dirichlet conditions (:func:`no_flow`, :func:`adjoint_constraints`). The
traction that holds each one comes back from the reaction there
(:func:`tractions`). Viscosity is tagged as the parameter "eta", so that::

    dJ = int K_eta deta dr,   K_eta = -2 r^2 B(u, u+),

is ``forward.kernel("eta", adjoint=adjoint_solution)``. The same equations,
typeset and worked through, are in ``tutorials/02_spherical_stokes.ipynb``.
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from ._typing import Coefficient
from .assembly import LoadTerm, Term
from .problem import Dirichlet, Solution

# A linear combination sum_k coeff_k(r) D^{d_k} field_k, as (coeff, field, d) triples.
Combination = list[tuple[Callable, str, int]]

ONE = lambda r: 1.0


def _as_coefficient(f: Coefficient | float):
    return f if callable(f) else (lambda r, c=float(f): c)


def _divergence(L: float):
    return [(ONE, "U", 1), (lambda r: 2 / r, "U", 0), (lambda r: -L / r, "V", 0)]


def _strains(L: float):
    """(weight, combination) pairs with B(a, b) = sum weight Q(a) Q(b)."""
    return [
        (1.0, [(ONE, "U", 1)]),
        (0.5, [(lambda r: 2 / r, "U", 0), (lambda r: -L / r, "V", 0)]),
        (L / 2, [(ONE, "V", 1), (lambda r: -1 / r, "V", 0), (lambda r: 1 / r, "U", 0)]),
        (L * (L - 2) / 2, [(lambda r: 1 / r, "V", 0)]),
        (-1 / 3, _divergence(L)),
    ]


def _pair(out, weight: Callable, trial: Combination, test: Combination, param=None):
    """Add int weight(r) trial * test dr to out, grouped by Term signature."""
    for c_a, f_a, d_a in trial:
        for c_b, f_b, d_b in test:
            out[(f_b, d_b, f_a, d_a, param)].append(
                lambda r, c_a=c_a, c_b=c_b: weight(r) * c_a(r) * c_b(r)
            )


def _terms(out):
    return [
        Term(lambda r, fs=fs: sum(f(r) for f in fs), d_test, d_trial, test, trial, param)
        for (test, d_test, trial, d_trial, param), fs in out.items()
    ]


def _viscous(out, L: float):
    for w, Q in _strains(L):
        _pair(out, lambda r, w=w: 2 * w * r**2, Q, Q, param="eta")


def forward_terms(l: int, rho0: Coefficient | float = 1.0, kappa: Coefficient | float = 0.0):
    """Bilinear form of the forward problem, viscosity tagged "eta".

    Parameters
    ----------
    l : int
        Spherical-harmonic degree (>= 1).
    rho0 : callable or float, default 1.0
        Reference density rho0(r) in the continuity equation.
    kappa : callable or float, default 0.0
        Anelastic coefficient kappa(r); 0 with constant rho0 is incompressible.

    Returns
    -------
    list of Term
        On fields "U", "V" and "P"; pass ``params={"eta": eta}`` to the Problem.

    Examples
    --------
    >>> terms = forward_terms(2)
    >>> sorted({t.test for t in terms}), {t.param for t in terms} == {"eta", None}
    (['P', 'U', 'V'], True)
    """
    L, rho0, kappa = l * (l + 1), _as_coefficient(rho0), _as_coefficient(kappa)
    div = _divergence(L)
    anelastic = div + [(lambda r: kappa(r) / r, "U", 0)]
    out = defaultdict(list)
    _viscous(out, L)  # int 2 eta r^2 B(u, u~)
    _pair(out, lambda r: -(r**2), [(ONE, "P", 0)], div)  # -int r^2 P div u~
    _pair(out, lambda r: -(r**2) * rho0(r), anelastic, [(ONE, "P", 0)])  # -int r^2 P~ rho0 [..]
    return _terms(out)


def adjoint_terms(l: int, rho0: Coefficient | float = 1.0, kappa: Coefficient | float = 0.0):
    """Bilinear form of the adjoint problem, viscosity tagged "eta".

    The trial fields are the adjoint unknowns (u+, P+) and the test fields the
    arbitrary fields (u~, P~), stored under the same names "U", "V" and "P".

    Parameters
    ----------
    l : int
        Spherical-harmonic degree (>= 1).
    rho0 : callable or float, default 1.0
        Reference density, as for :func:`forward_terms`.
    kappa : callable or float, default 0.0
        Anelastic coefficient, as for :func:`forward_terms`.

    Returns
    -------
    list of Term

    Examples
    --------
    Assembled, it is the transpose of the forward matrix:

    >>> import numpy as np
    >>> from fem_1d import FunctionSpace, assemble_system
    >>> s, p = FunctionSpace.build([0.5, 1.0], 3), FunctionSpace.build([0.5, 1.0], 2)
    >>> spaces, params = {"U": s, "V": s, "P": p}, {"eta": 1.0}
    >>> A, _ = assemble_system(spaces, forward_terms(2, kappa=0.5), params)
    >>> A_adj, _ = assemble_system(spaces, adjoint_terms(2, kappa=0.5), params)
    >>> bool(np.allclose(A_adj.toarray(), A.T.toarray()))
    True
    """
    L, rho0, kappa = l * (l + 1), _as_coefficient(rho0), _as_coefficient(kappa)
    div = _divergence(L)
    anelastic = div + [(lambda r: kappa(r) / r, "U", 0)]
    out = defaultdict(list)
    _viscous(out, L)  # int 2 eta r^2 B(u~, u+)
    _pair(out, lambda r: -(r**2), div, [(ONE, "P", 0)])  # -int r^2 P~ div u+
    _pair(out, lambda r: -(r**2) * rho0(r), [(ONE, "P", 0)], anelastic)  # -int r^2 P+ rho0 [..]
    return _terms(out)


def body_force(f: Coefficient, param: str | None = None):
    """The forward load int r^2 f(r) U~ dr.

    Parameters
    ----------
    f : callable
        Radial body force f(r), e.g. -g(r) drho(r).
    param : str, optional
        Parameter multiplying f, for density kernels.

    Returns
    -------
    list of LoadTerm

    Examples
    --------
    >>> body_force(lambda r: -10.0, param="drho")[0].test
    'U'
    """
    return [LoadTerm(lambda r: r**2 * f(r), test="U", param=param)]


def no_flow():
    """Forward conditions U(c) = U(b) = 0, held by the tractions T_c and T_b.

    Returns
    -------
    list of Dirichlet

    Examples
    --------
    >>> [d.at for d in no_flow()]
    ['left', 'right']
    """
    return [Dirichlet("U", "left"), Dirichlet("U", "right")]


def adjoint_constraints(j_b: float, j_c: float, b: float, c: float):
    """Adjoint conditions -b^2 U+(b) = j_b and c^2 U+(c) = j_c.

    Parameters
    ----------
    j_b, j_c : float
        Weights of the observable J = j_b T_b + j_c T_c.
    b, c : float
        Outer and inner radius.

    Returns
    -------
    list of Dirichlet

    Examples
    --------
    >>> [d.value for d in adjoint_constraints(1.0, 0.0, b=1.0, c=0.5)]
    [0.0, -1.0]
    """
    return [Dirichlet("U", "left", j_c / c**2), Dirichlet("U", "right", -j_b / b**2)]


def tractions(solution: Solution):
    """Boundary tractions (T_b, T_c) of a forward solution, or (lam_b, lam_c) of an adjoint one.

    Both problems have -b^2 T_b U~(b) + c^2 T_c U~(c) in the rows tested by
    U~, so the tractions are the reactions there scaled by -1/c^2 and 1/b^2.

    Parameters
    ----------
    solution : Solution
        Solution of a problem built with :func:`no_flow` or
        :func:`adjoint_constraints`.

    Returns
    -------
    (float, float)
        Traction at the outer and inner boundary.
    """
    xe = solution.problem.spaces["U"].mesh.xe
    c, b = xe[0], xe[-1]
    return solution.reaction("U", "right") / b**2, -solution.reaction("U", "left") / c**2
