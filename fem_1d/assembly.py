"""Weak-form assembly for one field or several coupled fields in 1D.

A bilinear form is a sum of Terms
    a(u, v) = sum_t  int c_t(x) d^{p_t} u / dx^{p_t} * d^{q_t} v / dx^{q_t} dx
with u a trial field and v a test field (possibly different fields and degrees).
A linear form is a sum of LoadTerms  l(v) = int f(x) d^q v / dx^q dx.

All terms are integrated with one GLL rule, of the highest degree among the
spaces, so a field integrated on its own GLL nodes gets a diagonal mass matrix.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._typing import (
    ArrayLike,
    Coefficient,
    FloatArray,
    IntArray,
    Location,
    Params,
    PointValue,
)
from .gll import ReferenceElement, basis_at, reference_element
from .space import FunctionSpace

DEFAULT_FIELD = "u"

# one space (field "u") or several named fields on the same mesh
Spaces = FunctionSpace | Mapping[str, FunctionSpace]


@dataclass(frozen=True)
class Term:
    """One term of a bilinear form: int coeff(x) * D^d_trial(trial) * D^d_test(test) dx.

    Parameters
    ----------
    coeff : callable
        ``coeff(x)`` with x an array of points, returning an array of the same
        shape or a scalar. Discontinuities are allowed at element boundaries.
    d_test, d_trial : {0, 1}, default 0
        Derivative order applied to the test and trial functions. Second
        derivatives are not supported: integrate by parts first.
    test, trial : str, default "u"
        Field names, matching keys of the spaces passed to
        :func:`assemble_system`.
    param : str, optional
        Name of a model parameter m(x) that multiplies the coefficient, so the
        term integrates ``coeff(x) * m(x)``. Tagging the parameter lets
        :meth:`Problem.kernel` compute sensitivities to it; the term must be
        linear in m.

    Examples
    --------
    The weak form of -(mu u')' + c u for one field:

    >>> import numpy as np
    >>> terms = [Term(lambda x: 1 + x, d_test=1, d_trial=1), Term(lambda x: 2.0)]

    A coupling block int q u' dx, testing field "p" with trial field "u":

    >>> coupling = Term(lambda x: 1.0, d_test=0, d_trial=1, test="p", trial="u")

    Viscous term 4 r^2 eta(r) U' V' with eta supplied later as a parameter:

    >>> viscous = Term(lambda r: 4 * r**2, 1, 1, test="U", trial="U", param="eta")
    """

    coeff: Coefficient
    d_test: int = 0
    d_trial: int = 0
    test: str = DEFAULT_FIELD
    trial: str = DEFAULT_FIELD
    param: str | None = None


@dataclass(frozen=True)
class LoadTerm:
    """One term of a linear form: int f(x) * D^d_test(test) dx.

    Parameters
    ----------
    f : callable
        ``f(x)`` with x an array of points, returning an array of the same shape
        or a scalar.
    d_test : {0, 1}, default 0
        Derivative order applied to the test function.
    test : str, default "u"
        Name of the field whose test functions this load acts on.
    param : str, optional
        Model parameter m(x) multiplying f, as for :class:`Term`.

    Examples
    --------
    >>> import numpy as np
    >>> load = LoadTerm(lambda x: np.sin(np.pi * x))

    A buoyancy load -g(r) r^2 rho(r) on the radial velocity, rho a parameter:

    >>> buoyancy = LoadTerm(lambda r: -10.0 * r**2, test="U", param="rho")
    """

    f: Coefficient
    d_test: int = 0
    test: str = DEFAULT_FIELD
    param: str | None = None


@dataclass(frozen=True)
class PointTerm:
    """A bilinear term at one point: value * trial(x0) * test(x0).

    These come from the boundary terms of integration by parts: Robin
    conditions, and couplings that act only at a boundary or interface (e.g.
    surface topography loading the gravitational potential).

    Parameters
    ----------
    value : float or callable
        Coefficient of the term, or ``value(params)`` computing it from the
        model parameters (not differentiated by :meth:`Problem.kernel`).
    at : "left", "right" or float
        Where the term acts; must be a node of both fields' spaces.
    test, trial : str, default "u"
        Field names, as for :class:`Term`.

    Examples
    --------
    The Robin condition u'(1) + 2 u(1) = g contributes 2 u(1) v(1) to the form:

    >>> robin = PointTerm(2.0, at="right")
    """

    value: PointValue
    at: Location
    test: str = DEFAULT_FIELD
    trial: str = DEFAULT_FIELD


@dataclass(frozen=True)
class PointLoad:
    """A load at one point: value * test(x0).

    Use it for Neumann and Robin data at a boundary, and for concentrated
    sources such as a surface density at an internal interface.

    Parameters
    ----------
    value : float or callable
        Size of the load, or ``value(params)`` computing it from the model
        parameters (not differentiated by :meth:`Problem.kernel`).
    at : "left", "right" or float
        Where the load acts; must be a node of the field's space.
    test : str, default "u"
        Field whose test functions the load acts on.

    Examples
    --------
    The Neumann condition u'(1) = 3 for -u'' = f adds 3 v(1) to the load:

    >>> flux = PointLoad(3.0, at="right")
    """

    value: PointValue
    at: Location
    test: str = DEFAULT_FIELD


def _as_spaces(spaces: Spaces):
    """Normalise a FunctionSpace or dict of them into a dict, checking the meshes."""
    if isinstance(spaces, FunctionSpace):
        return {DEFAULT_FIELD: spaces}
    spaces = dict(spaces)
    first, *rest = spaces.values()
    for s in rest:
        if not np.array_equal(s.mesh.xe, first.mesh.xe):
            raise ValueError(
                "all fields must share the same element end points"
            )
    return spaces


def _quadrature(spaces: Mapping[str, FunctionSpace]):
    """GLL rule of the highest degree among the spaces."""
    return reference_element(max(s.ref.n for s in spaces.values()))


def quadrature_points(space: FunctionSpace, quad: ReferenceElement):
    """Quadrature points and weights of every element.

    End points are nudged just inside the element, so at an interface each
    element sees its own side of a discontinuous coefficient.

    Parameters
    ----------
    space : FunctionSpace
        Any space on the mesh (only its element end points are used).
    quad : ReferenceElement
        GLL rule.

    Returns
    -------
    x : ndarray, shape (Ne, nq)
        Points.
    w : ndarray, shape (Ne, nq)
        Weights including the element Jacobian h / 2, so ``(w * f(x)).sum()``
        integrates f over the domain.

    Examples
    --------
    >>> from fem_1d import reference_element
    >>> s = FunctionSpace.build([0.0, 1.0, 3.0], 1)
    >>> x, w = quadrature_points(s, reference_element(2))
    >>> round(float(w.sum()), 12)
    3.0
    """
    xe, he = space.mesh.xe, space.mesh.he
    xq = xe[:-1, None] + 0.5 * (1 + quad.xi) * he[:, None]
    eps = 1e-12 * he
    xq[:, 0] += eps
    xq[:, -1] -= eps
    return xq, quad.weights * (he / 2)[:, None]


def _sample(f: Coefficient, space: FunctionSpace, quad: ReferenceElement):
    """f at the quadrature points of every element, shape (Ne, nq)."""
    xq, _ = quadrature_points(space, quad)
    return np.broadcast_to(f(xq), xq.shape).astype(float)


def field_at_quadrature(
    space: FunctionSpace, values: ArrayLike, quad: ReferenceElement, d: int = 0
):
    """A discrete field, or its x-derivative, at the quadrature points.

    Parameters
    ----------
    space : FunctionSpace
        Space of the field.
    values : ndarray, shape (n_dofs,)
        The field's dof values (e.g. ``solution["U"]``).
    quad : ReferenceElement
        GLL rule giving the points.
    d : {0, 1}, default 0
        0 for values, 1 for the x-derivative.

    Returns
    -------
    ndarray, shape (Ne, nq)
        Values at the points of :func:`quadrature_points`.

    Examples
    --------
    >>> import numpy as np
    >>> from fem_1d import reference_element
    >>> s = FunctionSpace.build([0.0, 0.5, 1.0], 2)
    >>> du = field_at_quadrature(s, s.x**2, reference_element(2), d=1)
    >>> bool(np.allclose(du, 2 * quadrature_points(s, reference_element(2))[0]))
    True
    """
    ue = np.asarray(values, dtype=float)[space.conn]  # (Ne, n + 1)
    out = ue @ _basis(space, quad, d).T
    return out * (2.0 / space.mesh.he[:, None]) ** d


def _with_param(f: Coefficient, param: str | None, params: Params | None):
    """f(x) times the named parameter, or f itself when untagged."""
    if param is None:
        return f
    if params is None or param not in params:
        raise KeyError(f"parameter {param!r} is not in params")
    m = params[param]
    m = m if callable(m) else (lambda x, c=float(m): c)
    return lambda x: f(x) * m(x)


def _value(v: PointValue, params: Params | None):
    """A point term's value, evaluated from the parameters if callable."""
    return float(v(params) if callable(v) else v)


def _basis(space: FunctionSpace, quad: ReferenceElement, d: int):
    """Basis (d = 0) or its xi-derivative (d = 1) at the quadrature nodes."""
    if d not in (0, 1):
        raise ValueError(
            "only first derivatives are supported (integrate by parts)"
        )
    return basis_at(space.ref, quad)[d]


def _scale(space: FunctionSpace, d: int):
    """Jacobian factor (2 / h)^d for d/dx and h / 2 for dx, shape (Ne,)."""
    return (2.0 / space.mesh.he) ** (d - 1)


def element_matrix(
    test_space: FunctionSpace,
    trial_space: FunctionSpace,
    coeff: Coefficient,
    d_test: int = 0,
    d_trial: int = 0,
    quad: ReferenceElement | None = None,
):
    """Element matrices of one bilinear term, before assembly.

    Parameters
    ----------
    test_space, trial_space : FunctionSpace
        Spaces of the test and trial functions (same element end points).
    coeff : callable
        Coefficient ``coeff(x)``, as in :class:`Term`.
    d_test, d_trial : {0, 1}, default 0
        Derivative orders on the test and trial functions.
    quad : ReferenceElement, optional
        GLL quadrature rule. Defaults to the higher of the two spaces' degrees.

    Returns
    -------
    Ae : ndarray, shape (Ne, n_test + 1, n_trial + 1)
        ``Ae[e, i, j]`` is the integral over element e with test basis i and
        trial basis j.

    Examples
    --------
    The degree-1 stiffness matrix on one unit element is [[1, -1], [-1, 1]]:

    >>> import numpy as np
    >>> s = FunctionSpace.build([0.0, 1.0], 1)
    >>> element_matrix(s, s, lambda x: 1.0, 1, 1)[0]
    array([[ 1., -1.],
           [-1.,  1.]])
    """
    quad = quad or reference_element(max(test_space.ref.n, trial_space.ref.n))
    c = _sample(coeff, test_space, quad)
    Bt, Bs = (
        _basis(test_space, quad, d_test),
        _basis(trial_space, quad, d_trial),
    )
    Ae = np.einsum("q,eq,qi,qj->eij", quad.weights, c, Bt, Bs)
    return Ae * _scale(test_space, d_test + d_trial)[:, None, None]


def element_load(
    space: FunctionSpace,
    f: Coefficient,
    d_test: int = 0,
    quad: ReferenceElement | None = None,
):
    """Element load vectors of one linear term, before assembly.

    Parameters
    ----------
    space : FunctionSpace
        Space of the test functions.
    f : callable
        Load ``f(x)``, as in :class:`LoadTerm`.
    d_test : {0, 1}, default 0
        Derivative order on the test function.
    quad : ReferenceElement, optional
        GLL quadrature rule. Defaults to the space's own element.

    Returns
    -------
    Fe : ndarray, shape (Ne, n + 1)
        ``Fe[e, i]`` is the integral of f times test basis i over element e.

    Examples
    --------
    A unit load on one unit element of degree 2 (Simpson's rule weights):

    >>> s = FunctionSpace.build([0.0, 1.0], 2)
    >>> element_load(s, lambda x: 1.0)[0]
    array([0.16666667, 0.66666667, 0.16666667])
    """
    quad = quad or space.ref
    Fe = np.einsum(
        "q,eq,qi->ei",
        quad.weights,
        _sample(f, space, quad),
        _basis(space, quad, d_test),
    )
    return Fe * _scale(space, d_test)[:, None]


def dof_slices(spaces: Spaces):
    """Where each field's degrees of freedom sit in the global vector.

    Parameters
    ----------
    spaces : FunctionSpace or dict of str to FunctionSpace
        One space (named "u") or several. Fields are stacked in dict order.

    Returns
    -------
    dict of str to slice
        ``x[slices[name]]`` is that field's part of a global vector x.

    Examples
    --------
    >>> xe = [0.0, 0.5, 1.0]
    >>> dof_slices({"u": FunctionSpace.build(xe, 3), "p": FunctionSpace.build(xe, 1)})
    {'u': slice(0, 7, None), 'p': slice(7, 10, None)}
    """
    out, start = {}, 0
    for name, s in _as_spaces(spaces).items():
        out[name] = slice(start, start + s.n_dofs)
        start += s.n_dofs
    return out


def _scatter(Ae: FloatArray, test_conn: IntArray, trial_conn: IntArray):
    """COO (rows, cols, values) for element matrices with the given connectivity."""
    rows = np.broadcast_to(test_conn[:, :, None], Ae.shape).ravel()
    cols = np.broadcast_to(trial_conn[:, None, :], Ae.shape).ravel()
    return rows, cols, Ae.ravel()


def assemble_system(
    spaces: Spaces,
    terms: Sequence[Term | PointTerm],
    params: Params | None = None,
):
    """Assemble a bilinear form into a global sparse matrix.

        Parameters
        ----------
        spaces : FunctionSpace or dict of str to FunctionSpace
            One space (field "u") or several fields on the same element end points.
            Fields are stacked in dict order (see :func:`dof_slices`).
        terms : sequence of Term or PointTerm
            Terms of the bilinear form. Terms on the same pair of fields add up.
        params : dict of str to callable or float, optional
            Model parameters m(x) for terms tagged with ``param``, and the
            argument passed to callable point-term values.

        Returns
        -------
        A : scipy.sparse.csr_array, shape (N, N)
            Global matrix; row = test dof, column = trial dof, N = total dofs.
        slices : dict of str to slice
            Each field's block of rows and columns, as from :func:`dof_slices`.

        Raises
        ------
        ValueError
            If the fields' meshes differ, a term asks for a derivative above 1, or
            a point term is not at a node.
        KeyError
            If a term names a field not in ``spaces``, or a parameter not in
            ``params``.

        Examples
        --------
        Solve -u'' = 2 with u(0) = u(1) = 0 by hand (what :class:`Problem` does
    for you); degree 2 reproduces the exact solution x(1 - x) to rounding:

        >>> import numpy as np
        >>> from scipy.sparse.linalg import spsolve
        >>> s = FunctionSpace.build(np.linspace(0, 1, 4), 2)
        >>> A, _ = assemble_system(s, [Term(lambda x: 1.0, 1, 1)])
        >>> b = assemble_load(s, [LoadTerm(lambda x: 2.0)])
        >>> inner = slice(1, -1)  # drop the fixed end dofs
        >>> u = np.zeros(s.n_dofs)
        >>> u[inner] = spsolve(A[inner, inner].tocsc(), b[inner])
        >>> bool(np.allclose(u, s.x * (1 - s.x)))
        True
    """
    spaces = _as_spaces(spaces)
    slices = dof_slices(spaces)
    quad = _quadrature(spaces)
    N = sum(s.n_dofs for s in spaces.values())
    rows, cols, vals = [], [], []
    for t in terms:
        te, tr = spaces[t.test], spaces[t.trial]
        if isinstance(t, PointTerm):
            rows.append([slices[t.test].start + te.dof_at(t.at)])
            cols.append([slices[t.trial].start + tr.dof_at(t.at)])
            vals.append([_value(t.value, params)])
            continue
        coeff = _with_param(t.coeff, t.param, params)
        Ae = element_matrix(te, tr, coeff, t.d_test, t.d_trial, quad)
        r, c, v = _scatter(
            Ae, te.conn + slices[t.test].start, tr.conn + slices[t.trial].start
        )
        rows.append(r), cols.append(c), vals.append(v)
    if not terms:
        return sparse.csr_array((N, N)), slices
    A = sparse.coo_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
    return A.tocsr(), slices


def assemble_load(
    spaces: Spaces,
    loads: Sequence[LoadTerm | PointLoad],
    params: Params | None = None,
):
    """Assemble a linear form into a global vector.

    Parameters
    ----------
    spaces : FunctionSpace or dict of str to FunctionSpace
        The same spaces as passed to :func:`assemble_system`.
    loads : sequence of LoadTerm or PointLoad
        Terms of the linear form.
    params : dict of str to callable or float, optional
        Model parameters, as for :func:`assemble_system`.

    Returns
    -------
    b : ndarray, shape (N,)
        Global load vector, fields stacked in dict order.

    Examples
    --------
    Since the basis functions sum to one, the entries of the load sum to the
    integral of f:

    >>> s = FunctionSpace.build([0.0, 0.5, 1.0], 3)
    >>> round(float(assemble_load(s, [LoadTerm(lambda x: x)]).sum()), 12)
    0.5
    """
    spaces = _as_spaces(spaces)
    slices = dof_slices(spaces)
    quad = _quadrature(spaces)
    b = np.zeros(sum(s.n_dofs for s in spaces.values()))
    for t in loads:
        s = spaces[t.test]
        if isinstance(t, PointLoad):
            b[slices[t.test].start + s.dof_at(t.at)] += _value(t.value, params)
            continue
        Fe = element_load(s, _with_param(t.f, t.param, params), t.d_test, quad)
        np.add.at(b, s.conn + slices[t.test].start, Fe)
    return b
