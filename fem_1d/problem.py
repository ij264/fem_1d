"""A linear boundary-value problem: assembled weak form plus Dirichlet conditions,
with forward and adjoint solves sharing one LU factorisation.

With dofs split into free (f) and Dirichlet (d), the solved system is

    A_ff u_f = b_f - A_fd u_d,

and the adjoint system uses the transpose of the same factors.

Sensitivity kernels follow from one forward and one adjoint solve. For a
functional J = c . u + r . (A u - b), with r on Dirichlet dofs (reactions),
and a parameter m entering A and b linearly,

    dJ = (lam - r) . (db - dA u),   A_ff^T lam_f = (c + A^T r)_f,

which, written as an integral over the domain, is dJ = int K(x) dm(x) dx.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import splu

from ._typing import (
    ArrayLike,
    Coefficient,
    FloatArray,
    Location,
    Params,
    PointValue,
    Side,
)
from .assembly import (
    DEFAULT_FIELD,
    LoadTerm,
    PointLoad,
    PointTerm,
    Spaces,
    Term,
    _as_spaces,
    _quadrature,
    _sample,
    _value,
    _with_param,
    assemble_load,
    assemble_system,
    dof_slices,
    field_at_quadrature,
    quadrature_points,
)


@dataclass(frozen=True)
class Dirichlet:
    """Fix one field's value at a point: u(x0) = value.

    Applied by eliminating that dof from the system. The force needed to hold
    it is available afterwards from :meth:`Solution.reaction`. Natural
    conditions (Neumann, Robin, free slip) are not constraints: they enter the
    weak form, as nothing or as :class:`PointTerm` and :class:`PointLoad`.

    Parameters
    ----------
    field : str, default "u"
        Name of the constrained field.
    at : "left", "right" or float, default "left"
        Where the value is imposed; must be a node of the field's space.
    value : float or callable, default 0.0
        Imposed value, or ``value(params)`` computing it from the parameters.

    Examples
    --------
    No radial flow through the bottom and top of the domain:

    >>> bcs = [Dirichlet("U", "left"), Dirichlet("U", "right")]
    """

    field: str = DEFAULT_FIELD
    at: Location = "left"
    value: PointValue = 0.0


class Problem:
    """Assemble a weak form with its Dirichlet conditions, ready to solve.

    Parameters
    ----------
    spaces : FunctionSpace or dict of str to FunctionSpace
        One space (field "u") or several fields on the same mesh.
    terms : sequence of Term or PointTerm
        Bilinear form, as for :func:`assemble_system`.
    loads : sequence of LoadTerm or PointLoad, default ()
        Linear form, as for :func:`assemble_load`.
    constraints : sequence of Dirichlet, default ()
        Essential conditions.
    params : dict of str to callable or float, optional
        Model parameters m(x) for terms and loads tagged with ``param``, also
        passed to callable point and Dirichlet values.

    Attributes
    ----------
    spaces : dict of str to FunctionSpace
        The fields, in stacking order.
    slices : dict of str to slice
        Each field's part of the global vector.
    A : scipy.sparse.csr_array, shape (N, N)
        Assembled matrix, before constraints are applied.
    b : ndarray, shape (N,)
        Assembled load vector.
    free : ndarray of bool, shape (N,)
        False at Dirichlet dofs.
    fixed : ndarray of int
        Indices of the Dirichlet dofs, increasing.
    fixed_values : ndarray
        Imposed value at each of ``fixed``.

    Raises
    ------
    ValueError
        If two Dirichlet conditions give different values to the same dof.
    TypeError
        If a constraint is not a :class:`Dirichlet`.

    Examples
    --------
    -u'' = 2 with u(0) = 1, u(1) = 0 has solution u = 1 - x^2, whose outward
    fluxes -u'(0) = 0 and u'(1) = -2 come back as reactions:

    >>> import numpy as np
    >>> from fem_1d import FunctionSpace, Term, LoadTerm
    >>> s = FunctionSpace.build(np.linspace(0, 1, 4), 2)
    >>> problem = Problem(s, [Term(lambda x: 1.0, 1, 1)], [LoadTerm(lambda x: 2.0)],
    ...                   [Dirichlet("u", "left", 1.0), Dirichlet("u", "right", 0.0)])
    >>> sol = problem.solve()
    >>> bool(np.allclose(sol["u"], 1 - s.x**2))
    True
    >>> abs(round(sol.reaction("u", "left"), 8)), round(sol.reaction("u", "right"), 8)
    (0.0, -2.0)
    """

    def __init__(
        self,
        spaces: Spaces,
        terms: Sequence[Term | PointTerm],
        loads: Sequence[LoadTerm | PointLoad] = (),
        constraints: Sequence[Dirichlet] = (),
        params: Params | None = None,
    ):
        self.spaces = _as_spaces(spaces)
        self.slices = dof_slices(self.spaces)
        self.terms, self.loads, self.params = list(terms), list(loads), params
        self.A, _ = assemble_system(self.spaces, self.terms, params)
        self.b = assemble_load(self.spaces, self.loads, params)

        fixed = {}
        for c in constraints:
            if not isinstance(c, Dirichlet):
                raise TypeError(f"constraints must be Dirichlet, got {type(c).__name__}")
            i, v = self.dof(c.field, c.at), _value(c.value, params)
            if i in fixed and fixed[i] != v:
                raise ValueError(f"conflicting Dirichlet values at dof {i} of field {c.field!r}")
            fixed[i] = v
        self.fixed = np.array(sorted(fixed), dtype=int)
        self.fixed_values = np.array([fixed[i] for i in self.fixed])
        self.free = np.ones(self.n_dofs, dtype=bool)
        self.free[self.fixed] = False
        self._lu = None

    @property
    def n_dofs(self):
        """int: Total number of dofs N, over all fields."""
        return self.A.shape[0]

    def dof(self, field: str, at: Location):
        """Global index of a field's dof at a point.

        Parameters
        ----------
        field : str
            Field name.
        at : "left", "right" or float
            Location, as for :meth:`FunctionSpace.dof_at`.

        Returns
        -------
        int
            Index into the global vector.
        """
        return self.slices[field].start + self.spaces[field].dof_at(at)

    def functional(
        self, field: str, x: Location, d: int = 0, side: Side = "above"
    ):
        """Weights c of the point functional J = u(x) (or u'(x)) on the global dofs.

        x need not be a node. Pass the result as ``c`` to :meth:`kernel` or
        :meth:`adjoint`. For a reaction functional, use ``r = functional(...)``
        at a Dirichlet dof instead.

        Parameters
        ----------
        field : str
            Field name.
        x : "left", "right" or float
            Point in the domain.
        d : {0, 1}, default 0
            0 for the value, 1 for the x-derivative.
        side : {"above", "below"}, default "above"
            Element used if x is on an element boundary.

        Returns
        -------
        ndarray, shape (N,)
        """
        s = self.spaces[field]
        x = {"left": s.mesh.xe[0], "right": s.mesh.xe[-1]}.get(x, x)
        c = np.zeros(self.n_dofs)
        c[self.slices[field]] = s.interpolation_matrix(x, d, side).toarray()[0]
        return c

    def _factors(self):
        """SuperLU factors of A_ff, computed once."""
        if self._lu is None:
            try:
                self._lu = splu(sparse.csc_matrix(self.A[self.free][:, self.free]))
            except RuntimeError as err:  # SuperLU reports singular matrices this way
                raise LinAlgError(str(err)) from err
        return self._lu

    def solve(self, b: ArrayLike | None = None):
        """Solve the forward problem.

        Parameters
        ----------
        b : ndarray, shape (N,), optional
            Load vector to use instead of the assembled one, e.g. to reuse the
            factors for another load.

        Returns
        -------
        Solution

        Raises
        ------
        scipy.linalg.LinAlgError
            If the constrained matrix is singular.
        """
        b = self.b if b is None else np.asarray(b, dtype=float)
        f, d = self.free, self.fixed
        u = np.zeros(self.n_dofs)
        u[d] = self.fixed_values
        u[f] = self._factors().solve(b[f] - self.A[f][:, d] @ u[d])
        return Solution(self, u, b)

    def adjoint(self, c: ArrayLike):
        """Solve the adjoint problem for the functional J(u) = c . u.

        With zero Dirichlet values, J = lam . b for the returned lam, so lam is
        the sensitivity of J to the load: one adjoint solve gives J's response
        to a load at every dof.

        Parameters
        ----------
        c : ndarray, shape (N,) or (N, k)
            Weights of the functional(s) on the global dofs. Several columns
            are solved together.

        Returns
        -------
        lam : ndarray, same shape as c
            Adjoint solution(s), zero at Dirichlet dofs.

        Examples
        --------
        J = u(0.5) for -u'' = f, u(0) = u(1) = 0; the adjoint is the Green's
        function, so J = lam . b for any load:

        >>> import numpy as np
        >>> from fem_1d import FunctionSpace, Term, LoadTerm
        >>> s = FunctionSpace.build(np.linspace(0, 1, 5), 2)
        >>> p = Problem(s, [Term(lambda x: 1.0, 1, 1)], [LoadTerm(np.sin)],
        ...             [Dirichlet("u", "left"), Dirichlet("u", "right")])
        >>> c = p.functional("u", 0.5)
        >>> bool(np.isclose(p.adjoint(c) @ p.b, c @ p.solve().u))
        True
        """
        c = np.asarray(c, dtype=float)
        lam = np.zeros_like(c)
        lam[self.free] = self._factors().solve(np.ascontiguousarray(c[self.free]), trans="T")
        return lam

    def kernel(
        self,
        param: str,
        c: ArrayLike | None = None,
        r: ArrayLike | None = None,
        relative: bool = False,
        solution: "Solution | None" = None,
        adjoint: "Solution | ArrayLike | None" = None,
    ):
        """Sensitivity kernel of a functional with respect to a model parameter.

        The functional is J(u) = c . u + r . (A u - b): a weighted sum of dof
        values, plus a weighted sum of reactions at Dirichlet dofs (e.g. the
        normal stress that gives dynamic topography). The kernel K satisfies
        dJ = int K(x) dm(x) dx for a perturbation dm of the parameter, exactly
        for the discrete problem (it is the derivative of the computed J, not of
        the continuous one).

        Instead of c and r, the adjoint can be given: the solution lam of an
        adjoint weak form assembled as its own :class:`Problem`, with the
        functional entering through its loads and Dirichlet values. Then
        dJ = l_dm(lam) - a_dm(u, lam), with a_dm and l_dm this problem's
        bilinear and linear forms restricted to the terms tagged ``param``.

        Parameters
        ----------
        param : str
            Name of the parameter, as used in ``Term.param``/``LoadTerm.param``.
        c : ndarray, shape (N,), optional
            Weights on the dof values (default none).
        r : ndarray, shape (N,), optional
            Weights on the reactions; must be zero at free dofs (default none).
        relative : bool, default False
            Return the kernel for d ln m instead, i.e. K(x) m(x), so that
            dJ = int K dm/m dx. Usual for viscosity.
        solution : Solution, optional
            Forward solution to reuse; solved if not given.
        adjoint : Solution or ndarray, shape (N,), optional
            Adjoint solution on the same fields and dofs as this problem,
            replacing c and r.

        Returns
        -------
        Kernel

        Raises
        ------
        ValueError
            If no term or load uses ``param``, neither c, r nor adjoint is
            given, adjoint is given with c or r or has the wrong size, or r is
            nonzero at a free dof.

        Examples
        --------
        Compliance J = b . u of -(mu u')' = 1 has kernel -(u')^2 with respect
        to mu (stiffening the string reduces its deflection):

        >>> import numpy as np
        >>> from fem_1d import FunctionSpace, Term, LoadTerm
        >>> s = FunctionSpace.build(np.linspace(0, 1, 4), 3)
        >>> p = Problem(s, [Term(lambda x: 1.0, 1, 1, param="mu")], [LoadTerm(lambda x: 1.0)],
        ...             [Dirichlet("u", "left"), Dirichlet("u", "right")], params={"mu": 1.0})
        >>> K = p.kernel("mu", c=p.b)
        >>> bool(np.allclose(K.values, -(0.5 - K.x) ** 2))  # u = x (1 - x) / 2
        True
        """
        uses = [t for t in self.terms + self.loads if isinstance(t, (Term, LoadTerm)) and t.param == param]
        if not uses:
            raise ValueError(f"no term or load depends on parameter {param!r}")
        if adjoint is not None:
            if c is not None or r is not None:
                raise ValueError("give the functional's weights c, r or its adjoint, not both")
            lam = np.asarray(adjoint.u if isinstance(adjoint, Solution) else adjoint, dtype=float)
            if lam.shape != (self.n_dofs,):
                raise ValueError(f"adjoint has shape {lam.shape}, expected ({self.n_dofs},)")
        elif c is None and r is None:
            raise ValueError("give the functional's weights c and/or r, or its adjoint")
        else:
            c = np.zeros(self.n_dofs) if c is None else np.asarray(c, dtype=float)
            r = np.zeros(self.n_dofs) if r is None else np.asarray(r, dtype=float)
            if np.any(r[self.free] != 0):
                raise ValueError("reaction weights r must be zero at free dofs")
            lam = self.adjoint(c + self.A.T @ r) - r

        sol = solution or self.solve()
        quad = _quadrature(self.spaces)
        x, w = quadrature_points(next(iter(self.spaces.values())), quad)

        def at(name: str, vec: FloatArray, d: int):
            return field_at_quadrature(self.spaces[name], vec[self.slices[name]], quad, d)

        K = np.zeros_like(x)
        for t in uses:
            if isinstance(t, Term):
                K -= _sample(t.coeff, self.spaces[t.test], quad) * at(t.test, lam, t.d_test) * at(t.trial, sol.u, t.d_trial)
            else:
                K += _sample(t.f, self.spaces[t.test], quad) * at(t.test, lam, t.d_test)
        if relative:
            K = K * _sample(_with_param(lambda x: 1.0, param, self.params), self.spaces[uses[0].test], quad)
        return Kernel(param=param, x=x, weights=w, values=K, relative=relative)


@dataclass(frozen=True)
class Kernel:
    """A sensitivity kernel sampled at the quadrature points.

    Attributes
    ----------
    param : str
        Parameter it is the sensitivity to.
    x : ndarray, shape (Ne, nq)
        Quadrature points (end points nudged inside each element, so a kernel
        that jumps at an interface has a value on each side).
    weights : ndarray, shape (Ne, nq)
        Quadrature weights including the element Jacobian.
    values : ndarray, shape (Ne, nq)
        Kernel K at the points.
    relative : bool
        True if K is with respect to ln m rather than m.

    Examples
    --------
    Plot with :func:`fem_1d.plotting.plot_kernel`, or
    ``plt.plot(K.x.ravel(), K.values.ravel())``.
    """

    param: str
    x: np.ndarray
    weights: np.ndarray
    values: np.ndarray
    relative: bool = False

    def apply(self, dm: Coefficient | float):
        """Predicted first-order change of the functional.

        Parameters
        ----------
        dm : callable or float
            Perturbation dm(x) of the parameter, or of ln m if the kernel is
            relative.

        Returns
        -------
        float
            int K(x) dm(x) dx, by the same quadrature used for assembly.
        """
        d = np.broadcast_to(dm(self.x) if callable(dm) else dm, self.x.shape)
        return float(np.sum(self.weights * self.values * d))


@dataclass(frozen=True)
class Solution:
    """Result of :meth:`Problem.solve`.

    Attributes
    ----------
    problem : Problem
        The problem that was solved.
    u : ndarray, shape (N,)
        Global solution, fields stacked as in ``problem.slices``.
    b : ndarray, shape (N,)
        Load vector that was used.

    Examples
    --------
    ``sol["U"]`` is the part of ``sol.u`` belonging to field "U".
    """

    problem: Problem
    u: np.ndarray
    b: np.ndarray

    def __getitem__(self, field: str):
        return self.u[self.problem.slices[field]]

    @property
    def fields(self):
        """dict of str to ndarray: Every field's part of ``u``."""
        return {name: self[name] for name in self.problem.slices}

    def evaluate(
        self, field: str, x: ArrayLike, d: int = 0, side: Side = "above"
    ):
        """A field of the solution, or its derivative, at arbitrary points.

        Parameters
        ----------
        field : str
            Field name.
        x : float or array_like
            Points in the domain.
        d : {0, 1}, default 0
            0 for values, 1 for the x-derivative.
        side : {"above", "below"}, default "above"
            Element used for points on element boundaries.

        Returns
        -------
        ndarray, shape (k,)
        """
        return self.problem.spaces[field].evaluate(self[field], x, d, side)

    def residual(self):
        """Global residual A u - b.

        Returns
        -------
        ndarray, shape (N,)
            Zero (to rounding) at free dofs; at Dirichlet dofs it is the
            reaction, see :meth:`reaction`.
        """
        return self.problem.A @ self.u - self.b

    def reaction(self, field: str, at: Location):
        """Force needed to hold a Dirichlet dof at its value.

        For a second-order term int mu u' v', the reaction at a boundary is the
        outward flux mu du/dn there: +mu u' at the right end, -mu u' at the
        left. For Stokes this is the normal stress on a no-flow boundary, which
        gives dynamic topography without differentiating the velocity.

        Parameters
        ----------
        field : str
            Field name.
        at : "left", "right" or float
            Location of the dof.

        Returns
        -------
        float
            Component of :meth:`residual` at that dof.
        """
        return float(self.residual()[self.problem.dof(field, at)])
