"""Gauss-Lobatto-Legendre (GLL) reference element on [-1, 1].

The interior nodes are computed as the roots of the Jacobi polynomial
P^(1,1)_(n-1), which is P_n' up to a constant, and the Lagrange basis is
evaluated off the nodes with the barycentric formula; both stay accurate to
rounding at high degree.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_legendre, roots_jacobi

from ._typing import ArrayLike, FloatArray


@dataclass(frozen=True)
class ReferenceElement:
    """GLL nodes, quadrature weights and derivative matrix on [-1, 1].

    Build one with :func:`reference_element` rather than directly.

    Attributes
    ----------
    n : int
        Polynomial degree; the element has n + 1 nodes.
    xi : ndarray, shape (n + 1,)
        GLL nodes in increasing order, including the end points -1 and 1.
    weights : ndarray, shape (n + 1,)
        GLL quadrature weights. They integrate polynomials of degree up to
        2n - 1 exactly on [-1, 1].
    D : ndarray, shape (n + 1, n + 1)
        Derivative matrix, ``D[q, k] = l_k'(xi_q)`` for the Lagrange basis
        l_k on the nodes. ``D @ f(xi)`` differentiates any polynomial f of
        degree <= n exactly.
    """

    n: int
    xi: np.ndarray
    weights: np.ndarray
    D: np.ndarray


def derivative_matrix(xi: FloatArray, n: int):
    """Derivative matrix of the Lagrange basis on the degree-n GLL nodes.

    Uses the closed form D[i, j] = P_n(xi_i) / (P_n(xi_j) (xi_i - xi_j)) for
    i != j. The diagonal is set to minus the sum of its row (the
    "negative sum trick" of Baltensperger & Trummer 2003), which is
    analytically the same (0 inside, -+n(n+1)/4 at the corners) but makes
    D annihilate constants to rounding and keeps D accurate at high degree:
    at n = 64 it differentiates polynomials ~30x more accurately than the
    exact diagonal.

    Parameters
    ----------
    xi : ndarray, shape (n + 1,)
        The degree-n GLL nodes (the formula is only valid for these).
    n : int
        Polynomial degree.

    Returns
    -------
    D : ndarray, shape (n + 1, n + 1)
        ``D[q, k] = l_k'(xi_q)``.

    Examples
    --------
    >>> import numpy as np
    >>> derivative_matrix(np.array([-1.0, 1.0]), 1)
    array([[-0.5,  0.5],
           [-0.5,  0.5]])
    """
    P = eval_legendre(n, xi)
    diff = xi[:, None] - xi[None, :]
    np.fill_diagonal(diff, 1.0)  # avoid dividing by zero
    D = (P[:, None] / P[None, :]) / diff
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D


def reference_element(n: int):
    """Build the degree-n GLL reference element.

    Parameters
    ----------
    n : int
        Polynomial degree, at least 1.

    Returns
    -------
    ReferenceElement
        Nodes, weights and derivative matrix for degree n.

    Raises
    ------
    ValueError
        If n < 1.

    Examples
    --------
    Degree 2 gives Simpson's rule on the nodes -1, 0, 1:

    >>> import numpy as np
    >>> ref = reference_element(2)
    >>> ref.weights
    array([0.33333333, 1.33333333, 0.33333333])
    >>> bool(np.allclose(ref.D @ ref.xi**2, 2 * ref.xi))  # d/dx x^2 = 2x
    True
    """
    if n < 1:
        raise ValueError("polynomial degree n must be >= 1")
    interior = np.sort(roots_jacobi(n - 1, 1.0, 1.0)[0]) if n > 1 else np.empty(0)
    xi = np.concatenate(([-1.0], interior, [1.0]))
    weights = 2.0 / (n * (n + 1) * eval_legendre(n, xi) ** 2)
    return ReferenceElement(n=n, xi=xi, weights=weights, D=derivative_matrix(xi, n))


def lagrange_basis(nodes: ArrayLike, x: ArrayLike):
    """Values of the Lagrange basis on ``nodes`` at arbitrary points.

    Uses the second barycentric formula, which is stable for any number of
    nodes; a point on (or within rounding of) a node gets that node's basis
    exactly.

    Parameters
    ----------
    nodes : array_like, shape (m,)
        Distinct interpolation nodes, e.g. ``ref.xi``.
    x : array_like, shape (k,)
        Evaluation points.

    Returns
    -------
    L : ndarray, shape (k, m)
        ``L[j, i] = l_i(x_j)``.

    Examples
    --------
    >>> import numpy as np
    >>> lagrange_basis([-1.0, 0.0, 1.0], [0.0, 0.5])
    array([[ 0.   ,  1.   ,  0.   ],
           [-0.125,  0.75 ,  0.375]])
    """
    nodes = np.asarray(nodes, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    w = 1.0 / diff.prod(axis=1)  # barycentric weights
    d = x[:, None] - nodes[None, :]
    on_node = np.abs(d) <= 4 * np.finfo(float).eps * max(1.0, np.abs(nodes).max())
    L = np.zeros((len(x), len(nodes)))
    hit = on_node.any(axis=1)
    L[hit] = on_node[hit]
    t = w[None, :] / d[~hit]
    L[~hit] = t / t.sum(axis=1, keepdims=True)
    return L


def basis_at(ref: ReferenceElement, quad: ReferenceElement):
    """Evaluate one element's Lagrange basis at another element's GLL nodes.

    Used to integrate fields of different degree with a shared quadrature rule.

    Parameters
    ----------
    ref : ReferenceElement
        Element whose basis functions l_j are evaluated (degree ref.n).
    quad : ReferenceElement
        Element whose GLL nodes xi_q are the evaluation points.

    Returns
    -------
    B : ndarray, shape (quad.n + 1, ref.n + 1)
        Basis values, B[q, j] = l_j(xi_q).
    Dxi : ndarray, shape (quad.n + 1, ref.n + 1)
        Derivatives on [-1, 1], Dxi[q, j] = l_j'(xi_q).

    Notes
    -----
    When ``ref.n == quad.n`` this returns exactly (I, ref.D), so a field sampled
    on its own GLL nodes keeps its diagonal mass matrix. Otherwise
    ``Dxi = B @ ref.D``: each l_j' has degree n - 1, so interpolating its
    nodal values is exact.

    Examples
    --------
    The degree-1 ("hat") basis at the three degree-2 nodes -1, 0, 1:

    >>> B, Dxi = basis_at(reference_element(1), reference_element(2))
    >>> B
    array([[1. , 0. ],
           [0.5, 0.5],
           [0. , 1. ]])
    >>> Dxi
    array([[-0.5,  0.5],
           [-0.5,  0.5],
           [-0.5,  0.5]])
    """
    if ref.n == quad.n:
        return np.eye(ref.n + 1), ref.D
    B = lagrange_basis(ref.xi, quad.xi)
    return B, B @ ref.D
