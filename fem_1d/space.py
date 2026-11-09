"""Function spaces: a mesh paired with the reference element of one field."""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ._typing import ArrayLike, Location, Side
from .gll import ReferenceElement, lagrange_basis, reference_element
from .mesh import Mesh


@dataclass(frozen=True)
class FunctionSpace:
    """Nodal (GLL) space of degree ``ref.n`` on a mesh, continuous or not.

    Fields of a coupled problem each get their own space on the same mesh, so
    their degrees may differ (e.g. velocity n, pressure n - 2), and so may
    their continuity: a discontinuous space gives every element its own dofs,
    so the field may jump at every element boundary. Put the element
    boundaries on the interfaces (see :func:`sized_mesh`) and a discontinuous
    pressure can jump where the viscosity does.

    Parameters
    ----------
    mesh : Mesh or array_like
        The mesh, or its element end points. Shared by all fields of a problem.
    ref : ReferenceElement
        Reference element giving the polynomial degree and GLL nodes.
    discontinuous : bool, default False
        Give each element its own end dofs instead of sharing them with its
        neighbours.

    Attributes
    ----------
    conn : ndarray of int, shape (Ne, n + 1)
        ``conn[e, k]`` is the dof index of local node k of element e.
    x : ndarray, shape (n_dofs,)
        Coordinate of each dof; in a discontinuous space every interior
        element boundary appears twice.

    Examples
    --------
    Two fields of different degree on the same three elements:

    >>> import numpy as np
    >>> xe = np.linspace(0.0, 1.0, 4)
    >>> u, p = FunctionSpace.build(xe, 4), FunctionSpace.build(xe, 2, discontinuous=True)
    >>> u.n_dofs, p.n_dofs
    (13, 9)
    >>> p.conn
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    """

    mesh: Mesh
    ref: ReferenceElement
    discontinuous: bool = False
    conn: np.ndarray = field(init=False, repr=False, compare=False)
    x: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.mesh, Mesh):
            object.__setattr__(self, "mesh", Mesh(self.mesh))
        xe, he, n = self.mesh.xe, self.mesh.he, self.ref.n
        Ne = len(he)
        stride = n + 1 if self.discontinuous else n  # elements share an end dof unless broken
        conn = np.arange(Ne)[:, None] * stride + np.arange(n + 1)
        x = np.empty(conn[-1, -1] + 1)
        x[conn] = xe[:-1, None] + 0.5 * (1 + self.ref.xi) * he[:, None]
        object.__setattr__(self, "conn", conn)
        object.__setattr__(self, "x", x)

    @classmethod
    def build(cls, mesh: Mesh | ArrayLike, n: int, discontinuous: bool = False):
        """Space of degree n on a mesh.

        Parameters
        ----------
        mesh : Mesh or array_like, shape (Ne + 1,)
            The mesh, or its strictly increasing element end points.
        n : int
            Polynomial degree (>= 1).
        discontinuous : bool, default False
            As for the class.

        Returns
        -------
        FunctionSpace
        """
        return cls(mesh, reference_element(n), discontinuous)

    @property
    def n_dofs(self):
        """int: Number of degrees of freedom."""
        return len(self.x)

    @property
    def boundary_dofs(self):
        """ndarray of int: Dofs on element boundaries, in order (both of a pair if discontinuous).

        Examples
        --------
        >>> FunctionSpace.build([0.0, 0.5, 1.0], 3).boundary_dofs
        array([0, 3, 6])
        """
        return np.unique(np.r_[self.conn[:, 0], self.conn[:, -1]])

    def dof_at(self, at: Location):
        """Index of the dof at a point.

        Parameters
        ----------
        at : "left", "right" or float
            The left or right end of the domain, or a coordinate, which must be
            a node of this space (to within 1e-10 of the domain length).

        Returns
        -------
        int
            Dof index into this space's vector (not the global system).

        Raises
        ------
        ValueError
            If no node lies at ``at``, or two do (an interior element boundary
            of a discontinuous space).

        Examples
        --------
        >>> s = FunctionSpace.build([0.0, 0.5, 1.0], 2)
        >>> s.dof_at("left"), s.dof_at(0.25), s.dof_at(0.5), s.dof_at("right")
        (0, 1, 2, 4)
        """
        if isinstance(at, str):
            if at == "left":
                return 0
            if at == "right":
                return self.n_dofs - 1
            raise ValueError(f'at must be "left", "right" or a coordinate, got {at!r}')
        i = np.flatnonzero(np.abs(self.x - at) <= 1e-10 * self.mesh.length)
        if len(i) != 1:
            raise ValueError(
                f"no node of this space at {at}" if not len(i)
                else f"{at} is an element boundary of a discontinuous space, which has two dofs there"
            )
        return int(i[0])

    def interpolation_matrix(self, x: ArrayLike, d: int = 0, side: Side = "above"):
        """Sparse matrix that evaluates a field (or its derivative) at points.

        Parameters
        ----------
        x : float or array_like, shape (k,)
            Points in the domain; need not be nodes.
        d : {0, 1}, default 0
            0 for values, 1 for the x-derivative.
        side : {"above", "below"}, default "above"
            Which element a point on an element boundary is evaluated in. It
            matters for derivatives (which jump between elements) and for
            discontinuous fields; values of a continuous field agree on both
            sides.

        Returns
        -------
        scipy.sparse.csr_array, shape (k, n_dofs)
            ``P @ values`` gives the field at the points. A row is also the
            weight vector ``c`` of the point functional J = u(x) (or u'(x))
            for :meth:`Problem.kernel`.

        Raises
        ------
        ValueError
            If d is not 0 or 1, a point is outside the domain, or side is
            invalid.

        Examples
        --------
        >>> import numpy as np
        >>> s = FunctionSpace.build([0.0, 0.5, 1.0], 3)
        >>> P = s.interpolation_matrix([0.1, 0.75], d=1)
        >>> P.shape, bool(np.allclose(P @ s.x**3, 3 * np.array([0.1, 0.75]) ** 2))
        ((2, 7), True)
        """
        if d not in (0, 1):
            raise ValueError("d must be 0 or 1")
        x = np.atleast_1d(np.asarray(x, dtype=float))
        e = np.atleast_1d(self.mesh.element_of(x, side))
        xe, he = self.mesh.xe, self.mesh.he
        xi = np.clip(2 * (x - xe[e]) / he[e] - 1, -1.0, 1.0)
        W = lagrange_basis(self.ref.xi, xi)
        if d:
            W = (W @ self.ref.D) * (2.0 / he[e])[:, None]
        rows = np.repeat(np.arange(len(x)), self.ref.n + 1)
        return sparse.csr_array((W.ravel(), (rows, self.conn[e].ravel())), shape=(len(x), self.n_dofs))

    def evaluate(
        self, values: ArrayLike, x: ArrayLike, d: int = 0, side: Side = "above"
    ):
        """A field's value (or derivative) at arbitrary points.

        Parameters
        ----------
        values : ndarray, shape (n_dofs,)
            The field's dof values.
        x : float or array_like, shape (k,)
            Points in the domain.
        d : {0, 1}, default 0
            0 for values, 1 for the x-derivative.
        side : {"above", "below"}, default "above"
            Element used for points on element boundaries (see
            :meth:`interpolation_matrix`).

        Returns
        -------
        ndarray, shape (k,)

        Examples
        --------
        A discontinuous field takes the value of the requested side at an
        element boundary:

        >>> import numpy as np
        >>> s = FunctionSpace.build([0.0, 0.5, 1.0], 1, discontinuous=True)
        >>> u = np.array([0.0, 1.0, 5.0, 6.0])
        >>> s.evaluate(u, [0.25, 0.5]), s.evaluate(u, 0.5, side="below")
        (array([0.5, 5. ]), array([1.]))
        """
        return self.interpolation_matrix(x, d, side) @ np.asarray(values, dtype=float)
