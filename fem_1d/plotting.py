"""Figures for meshes, reference elements and sensitivity kernels.

Every function takes arrays, builds its figure under STYLE, and returns it;
saving and showing are left to the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ._typing import ArrayLike, FloatArray
from .assembly import Spaces
from .gll import ReferenceElement, lagrange_basis
from .problem import Kernel
from .space import FunctionSpace

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Categorical slots in fixed order (validated for adjacent-pair colour-blind separation).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
}
MARKER = dict(marker="o", ms=6, mec=SURFACE, mew=1.5)

_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "kappa",
          "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega"}


def _series_colors(k: int):
    """First k categorical colours; refuses more than the palette holds."""
    if k > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series per panel; split into small multiples")
    return SERIES[:k]


def plot_mesh(
    spaces: Spaces,
    interfaces: Sequence[float] = (),
    title: str = "Mesh and GLL nodes",
    figsize: tuple[float, float] | None = None,
):
    """Every field's nodes on the shared mesh, one row per field.

    Element boundaries are drawn as faint vertical lines through all rows and
    the given interfaces as dashed ones. For a discontinuous field, the two
    dofs at each interior element boundary are drawn as a pair of half-filled
    markers nudged apart, so the duplication is visible.

    Parameters
    ----------
    spaces : FunctionSpace or dict of str to FunctionSpace
        One space (labelled "u") or several fields on the same mesh; at most 4
        (one per palette colour), drawn top to bottom in dict order.
    interfaces : sequence of float, default ()
        Physical interfaces to highlight (e.g. the 660 km discontinuity).
    title : str, default "Mesh and GLL nodes"
        Axes title.
    figsize : tuple of float, optional
        Figure size; defaults to a width of 10 and 0.7 per field.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If more than 4 fields are given.

    Examples
    --------
    >>> mesh = sized_mesh([0.546, 0.896, 1.0], drmax=0.05)  # doctest: +SKIP
    >>> fig = plot_mesh({"U": FunctionSpace.build(mesh, 6),
    ...                  "P": FunctionSpace.build(mesh, 4, discontinuous=True)},
    ...                 interfaces=[0.896])  # doctest: +SKIP
    """
    spaces = {"u": spaces} if isinstance(spaces, FunctionSpace) else dict(spaces)
    colors = _series_colors(len(spaces))
    first = next(iter(spaces.values()))
    xe = first.mesh.xe
    nudge = 0.004 * (xe[-1] - xe[0])
    rows = len(spaces)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=figsize or (10, 1.0 + 0.7 * rows), constrained_layout=True)
        ax.grid(False)
        for x in xe:
            ax.axvline(x, color=GRID, lw=1, zorder=0)
        for x in interfaces:
            ax.axvline(x, color=INK_2, lw=1.2, ls="--", zorder=1)
        labels = []
        for row, ((name, s), color) in enumerate(zip(spaces.items(), colors)):
            y = rows - 1 - row
            ax.hlines(y, xe[0], xe[-1], color=GRID, lw=2, zorder=1)
            paired = np.zeros(s.n_dofs, dtype=bool)
            if s.discontinuous:
                paired[s.conn[:-1, -1]] = paired[s.conn[1:, 0]] = True
            ax.plot(s.x[~paired], np.full((~paired).sum(), y), lw=0, color=color, zorder=3,
                    marker="o", ms=5, mec=SURFACE, mew=1)
            for dofs, shift, fill in ((s.conn[:-1, -1], -nudge, "left"), (s.conn[1:, 0], nudge, "right")):
                if s.discontinuous:
                    ax.plot(s.x[dofs] + shift, np.full(len(dofs), y), lw=0, color=color, zorder=3,
                            marker="o", ms=7, fillstyle=fill, markerfacecoloralt=SURFACE, mec=color, mew=1)
            labels.append(f"{name}   degree {s.ref.n}, {s.n_dofs} dofs")
        ax.set_yticks(range(rows - 1, -1, -1), labels)
        ax.set_ylim(-0.6, rows - 0.4)
        ax.set_xlim(xe[0] - 0.01 * (xe[-1] - xe[0]), xe[-1] + 0.01 * (xe[-1] - xe[0]))
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("$x$")
        ax.set_title(f"{title}: {first.mesh.n_elements} elements")
    return fig


def plot_basis(ref: ReferenceElement, highlight: int | None = None):
    """The Lagrange basis and GLL weights of a reference element.

    Parameters
    ----------
    ref : ReferenceElement
        Element to draw.
    highlight : int, optional
        Index of one basis function to draw in colour (default the middle
        one); the others are drawn thin and grey.

    Returns
    -------
    matplotlib.figure.Figure
        Left: the n + 1 basis functions on [-1, 1], each 1 at its own node and
        0 at the others. Right: the quadrature weights at the nodes.

    Examples
    --------
    >>> fig = plot_basis(reference_element(4))  # doctest: +SKIP
    """
    k = ref.n // 2 if highlight is None else highlight
    xi = np.linspace(-1, 1, 401)
    L = lagrange_basis(ref.xi, xi)
    with plt.style.context(STYLE):
        fig, (ax_b, ax_w) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [2.2, 1]})
        ax_b.axhline(0, color=GRID, lw=1)
        for j in range(ref.n + 1):
            if j != k:
                ax_b.plot(xi, L[:, j], color=INK_2, lw=1, alpha=0.45)
        ax_b.plot(xi, L[:, k], color=BLUE, label=rf"$\ell_{{{k}}}(\xi)$")
        ax_b.plot(ref.xi, np.zeros_like(ref.xi), lw=0, color=INK, marker="o", ms=5, mec=SURFACE, mew=1)
        ax_b.plot(ref.xi[k], 1.0, lw=0, color=BLUE, **MARKER)
        ax_b.set(xlabel=r"$\xi$", ylabel="basis value", xlim=(-1.05, 1.05))
        ax_b.legend(loc="upper right")
        ax_b.set_title(f"Lagrange basis on the degree-{ref.n} GLL nodes")

        ax_w.vlines(ref.xi, 0, ref.weights, color=BLUE, lw=2)
        ax_w.plot(ref.xi, ref.weights, lw=0, color=BLUE, **MARKER)
        ax_w.set(xlabel=r"$\xi$", ylabel=r"weight $w_q$", xlim=(-1.1, 1.1), ylim=(0, None))
        ax_w.set_title(f"Weights (sum {ref.weights.sum():.0f})")
    return fig


def _tex_symbol(name: str):
    """A parameter name as TeX: Greek names become symbols, others upright text."""
    return f"\\{name}" if name in _GREEK else f"\\mathrm{{{name}}}"


def plot_kernel(
    kernel: Kernel,
    exact: Callable[[FloatArray], ArrayLike] | None = None,
    ax: Axes | None = None,
    label: str = "kernel",
    title: str | None = None,
):
    """A sensitivity kernel element by element, so jumps between elements show.

    Parameters
    ----------
    kernel : Kernel
        From :meth:`Problem.kernel`.
    exact : callable, optional
        Exact kernel ``exact(x)``, drawn dashed for comparison.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (styled by the caller); a new figure if omitted.
    label : str, default "kernel"
        Legend label of the computed kernel.
    title : str, optional
        Axes title (default names the parameter).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> K = problem.kernel("mu", c=problem.functional("u", 0.5))  # doctest: +SKIP
    >>> fig = plot_kernel(K)  # doctest: +SKIP
    """
    with plt.style.context(STYLE):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
        else:
            fig = ax.figure
        ax.axhline(0, color=GRID, lw=1)
        for e, (x, v) in enumerate(zip(kernel.x, kernel.values)):
            ax.plot(x, v, color=BLUE, label=label if e == 0 else None)
        if exact is not None:
            xx = np.linspace(kernel.x.min(), kernel.x.max(), 801)
            ax.plot(xx, exact(xx), color=INK_2, lw=1.5, ls="--", label="exact")
        ax.set_xlabel("$x$")
        p = _tex_symbol(kernel.param)
        ax.set_ylabel(f"$K_{{\\ln {p}}}$" if kernel.relative else f"$K_{{{p}}}$")
        ax.set_title(title or f"Sensitivity kernel for ${_tex_symbol(kernel.param)}$")
        ax.legend(loc="best")
    return fig
