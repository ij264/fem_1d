"""1D meshes: element end points only.

Nodes and dof numbering depend on each field's degree and continuity, so they
live in FunctionSpace, not here.
"""

from dataclasses import dataclass, field

import numpy as np

from ._typing import ArrayLike, Side


@dataclass(frozen=True)
class Mesh:
    """Elements of a 1D domain, given by their end points.

    Parameters
    ----------
    xe : array_like, shape (Ne + 1,)
        Element end points, strictly increasing. The domain is [xe[0], xe[-1]].

    Attributes
    ----------
    xe : ndarray, shape (Ne + 1,)
        Element end points.
    he : ndarray, shape (Ne,)
        Element lengths.

    Raises
    ------
    ValueError
        If xe has fewer than two points or is not strictly increasing.

    Examples
    --------
    >>> mesh = Mesh([0.0, 1.0, 3.0])
    >>> mesh.n_elements, mesh.he
    (2, array([1., 2.]))
    """

    xe: np.ndarray
    he: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        xe = np.asarray(self.xe, dtype=float)
        if xe.ndim != 1 or len(xe) < 2:
            raise ValueError("need at least two element end points")
        he = np.diff(xe)
        if np.any(he <= 0):
            raise ValueError("element end points must be strictly increasing")
        object.__setattr__(self, "xe", xe)
        object.__setattr__(self, "he", he)

    def __eq__(self, other: object):
        return isinstance(other, Mesh) and np.array_equal(self.xe, other.xe)

    def __hash__(self):
        return hash(self.xe.tobytes())

    @property
    def n_elements(self):
        """int: Number of elements Ne."""
        return len(self.he)

    @property
    def length(self):
        """float: Length of the domain."""
        return float(self.xe[-1] - self.xe[0])

    def element_of(self, x: ArrayLike, side: Side = "above"):
        """Index of the element containing each point.

        Parameters
        ----------
        x : float or array_like
            Points in the domain (to within 1e-10 of its length).
        side : {"above", "below"}, default "above"
            Which element a point on an interior element boundary belongs to:
            the one at larger x ("above") or smaller x ("below"). The domain
            ends always resolve to the first and last elements.

        Returns
        -------
        int or ndarray of int
            Element index, with the shape of ``x``.

        Raises
        ------
        ValueError
            If a point lies outside the domain, or ``side`` is invalid.

        Examples
        --------
        >>> mesh = Mesh([0.0, 1.0, 2.0, 3.0])
        >>> mesh.element_of([0.5, 1.0, 3.0]), mesh.element_of(1.0, side="below")
        (array([0, 1, 2]), 0)
        """
        if side not in ("above", "below"):
            raise ValueError(f'side must be "above" or "below", got {side!r}')
        x = np.asarray(x, dtype=float)
        tol = 1e-10 * self.length
        if np.any((x < self.xe[0] - tol) | (x > self.xe[-1] + tol)):
            raise ValueError(f"points must lie in [{self.xe[0]:g}, {self.xe[-1]:g}]")
        # snap points within tol of a boundary onto it, so the side rule applies
        k = np.clip(np.searchsorted(self.xe, x), 0, len(self.xe) - 1)
        near = np.abs(self.xe[k] - x) <= tol
        k_low = np.clip(k - 1, 0, None)
        near_low = np.abs(self.xe[k_low] - x) <= tol
        xs = np.where(near, self.xe[k], np.where(near_low, self.xe[k_low], x))
        e = np.searchsorted(self.xe, xs, side="right" if side == "above" else "left") - 1
        e = np.clip(e, 0, self.n_elements - 1)
        return int(e) if e.ndim == 0 else e

    def truncated(self, x: float):
        """The mesh from the element containing x upwards.

        Used to skip the deep interior of a radial problem where a degree-l
        solution is negligible (see :func:`truncation_radius`). The new left
        end is an element boundary at or below x, and is not a physical
        boundary: impose whatever condition stands in for the removed part
        (e.g. a vanishing solution) there.

        Parameters
        ----------
        x : float
            Point that must stay inside the mesh; values at or below the left
            end return the mesh unchanged.

        Returns
        -------
        Mesh

        Examples
        --------
        >>> Mesh([0.0, 1.0, 2.0, 3.0]).truncated(1.5).xe
        array([1., 2., 3.])
        """
        if x <= self.xe[0]:
            return self
        return Mesh(self.xe[self.element_of(min(x, self.xe[-1])):])


def uniform_mesh(a: float, Ne: int):
    """Ne equal elements on [0, a].

    Parameters
    ----------
    a : float
        Length of the domain [0, a].
    Ne : int
        Number of elements.

    Returns
    -------
    Mesh

    Examples
    --------
    >>> uniform_mesh(1.0, 4).xe
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    return Mesh(np.linspace(0.0, a, Ne + 1))


def sized_mesh(
    breakpoints: ArrayLike,
    drmax: float | None = None,
    lmax: int | None = None,
    per_degree: float = 0.1,
):
    """Mesh with given breakpoints, each span cut into elements no wider than drmax.

    Every breakpoint (domain ends and interfaces) is an element boundary, and
    each span between consecutive breakpoints is divided uniformly into
    ``ceil(span / drmax)`` elements. For spherical problems the element size
    can instead follow the largest harmonic degree to be resolved.

    Parameters
    ----------
    breakpoints : array_like, shape (k,)
        Strictly increasing points: the domain ends plus any interfaces, e.g.
        ``[r_cmb, r_660, r_surface]``. Need not start at 0.
    drmax : float, optional
        Largest element width.
    lmax : int, optional
        Largest spherical-harmonic degree; sets
        ``drmax = per_degree * x_max / (lmax + 1)`` with x_max the outer
        breakpoint (the rule used by planetmodel's RadialMesh).
    per_degree : float, default 0.1
        Element width per unit of x_max / (lmax + 1), for ``lmax``.

    Returns
    -------
    Mesh

    Raises
    ------
    ValueError
        If not exactly one of drmax and lmax is given, either is
        non-positive (lmax negative), or the breakpoints are not increasing.

    Examples
    --------
    A mantle shell from the core-mantle boundary to the surface, in units of
    the planet's radius, with the 660 km discontinuity pinned:

    >>> mesh = sized_mesh([0.546, 0.896, 1.0], drmax=0.1)
    >>> mesh.n_elements, bool((mesh.he <= 0.1).all())
    (6, True)
    """
    if (drmax is None) == (lmax is None):
        raise ValueError("give exactly one of drmax and lmax")
    b = np.asarray(breakpoints, dtype=float)
    if b.ndim != 1 or len(b) < 2 or np.any(np.diff(b) <= 0):
        raise ValueError("breakpoints must be at least two strictly increasing points")
    if lmax is not None:
        if lmax < 0:
            raise ValueError("lmax must be non-negative")
        drmax = per_degree * abs(b[-1]) / (lmax + 1)
    if drmax <= 0:
        raise ValueError("drmax must be positive")
    pieces = [np.linspace(lo, hi, max(1, int(np.ceil((hi - lo) / drmax))) + 1)[:-1]
              for lo, hi in zip(b[:-1], b[1:])]
    return Mesh(np.concatenate(pieces + [b[-1:]]))


def truncation_radius(l: int, x_max: float, eps: float = 1e-8):
    """Radius below which a degree-l interior solution is negligible.

    Solutions of Laplace-like radial problems at degree l decay inwards like
    (r / x_max)^(l + 1), so below the returned radius they are smaller than
    eps relative to their value at x_max.

    Parameters
    ----------
    l : int
        Spherical-harmonic degree (>= 0).
    x_max : float
        Outer radius.
    eps : float, default 1e-8
        Relative size considered negligible.

    Returns
    -------
    float
        ``x_max * eps ** (1 / (l + 1))``.

    Raises
    ------
    ValueError
        If l < 0 or eps is not in (0, 1).

    Examples
    --------
    At degree 100 the outer quarter of the radius carries everything above
    eps = 1e-8:

    >>> round(truncation_radius(100, 1.0), 3)
    0.833
    """
    if l < 0:
        raise ValueError("degree must be non-negative")
    if not 0 < eps < 1:
        raise ValueError("eps must be in (0, 1)")
    return float(x_max * eps ** (1.0 / (l + 1)))
