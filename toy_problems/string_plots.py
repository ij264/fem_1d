"""Figures for the string's normal modes and wave propagation (used by waves_on_string.py)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

from fem_1d.plotting import BLUE, GRID, INK_2, MARKER, ORANGE, STYLE, SURFACE, SERIES, _series_colors

def resolved_modes(rel_err, tol):
    """Number of leading modes resolved to within a tolerance.

    Parameters
    ----------
    rel_err : ndarray, shape (k,)
        Relative frequency error of each mode, lowest mode first.
    tol : float
        Error threshold.

    Returns
    -------
    int
        Count of modes before the first one whose error exceeds tol (all k
        if none does).

    Examples
    --------
    >>> import numpy as np
    >>> resolved_modes(np.array([1e-8, 1e-6, 1e-2, 1e-7]), tol=1e-3)
    2
    """
    bad = np.flatnonzero(rel_err > tol)
    return int(bad[0]) if bad.size else len(rel_err)


def plot_frequencies(omega, omega_exact, nodes_per_wavelength, title="", tol=1e-3):
    """Discrete against exact frequencies, and their relative error.

    Two panels share a mode-number axis, with nodes per wavelength on a second
    axis along the top. Modes past the first one above ``tol`` are shaded as
    unresolved.

    Parameters
    ----------
    omega : ndarray, shape (k,)
        Discrete frequencies, lowest first.
    omega_exact : ndarray, shape (k,)
        Exact frequencies of the same modes.
    nodes_per_wavelength : callable
        Maps a mode number p (float array) to nodes per wavelength; must be
        monotone so it can label the top axis. For a uniform string of N nodes,
        ``lambda p: 2 * N / p``.
    title : str, default ""
        Appended to the error panel's title.
    tol : float, default 1e-3
        Relative error that counts as resolved.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_frequencies(omega, np.pi * p, lambda q: 2 * N / q)  # doctest: +SKIP
    >>> fig.savefig("frequencies.png")  # doctest: +SKIP
    """
    p = np.arange(1, len(omega) + 1)
    rel_err = np.abs(omega - omega_exact) / omega_exact
    p_res = resolved_modes(rel_err, tol)

    with plt.style.context(STYLE):
        fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        ax_f.plot(p, omega_exact, color=INK_2, lw=1.5, ls="--", label=r"Exact $p\pi c/a$")
        ax_f.plot(p, omega, color=BLUE, lw=0, label="FEM", **MARKER)
        ax_f.set(xlabel="Mode number $p$", ylabel=r"Angular frequency $\omega_p$")
        ax_f.legend(loc="upper left")

        ax_e.semilogy(p, rel_err, color=BLUE, **MARKER)
        ax_e.set(xlabel="Mode number $p$", ylabel=r"$|\omega_p - p\pi c/a|\,/\,(p\pi c/a)$")
        ax_e.axhline(tol, color=INK_2, lw=1, ls=":")
        ax_e.text(p[0], tol * 1.3, f"tolerance {tol:g}", color=INK_2, fontsize=8)

        npw = lambda q: nodes_per_wavelength(np.maximum(q, 1e-9))
        for ax in (ax_f, ax_e):
            if p_res < len(p):
                ax.axvspan(p_res + 0.5, p[-1] + 0.5, color=ORANGE, alpha=0.10, lw=0)
            ax.set_xlim(0.5, p[-1] + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            top = ax.secondary_xaxis("top", functions=(npw, npw))
            top.set_xlabel("Nodes per wavelength", fontsize=9)
            top.set_xticks([t for t in (20, 10, 8, 6, 5, 4, 3) if npw(p[0]) >= t >= npw(p[-1])])
            top.spines["top"].set_visible(False)
        if p_res < len(p):
            ax_f.text(p_res + 1, ax_f.get_ylim()[0], "unresolved", color=ORANGE,
                      fontsize=9, va="bottom", fontweight="bold")

        ax_f.set_title("Eigenfrequencies", pad=34)
        ax_e.set_title(f"Relative error {title}".strip(), pad=34)
    return fig


def plot_modes(x, omega, modes, exact_shape, element_nodes, n_modes=4):
    """Small multiples of the first mode shapes against exact ones.

    Each FEM mode is flipped to match the sign of the exact shape and scaled
    to unit peak, since eigenvectors have arbitrary sign and normalisation.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Node coordinates.
    omega : ndarray, shape (k,)
        Frequencies, shown in the panel titles.
    modes : ndarray, shape (N, k)
        Mode shapes as columns, k >= n_modes.
    exact_shape : callable
        ``exact_shape(p, x)`` gives the exact mode p (1-based) at points x,
        with unit peak.
    element_nodes : ndarray of int
        Indices of nodes on element boundaries, marked on each curve.
    n_modes : int, default 4
        Number of modes to plot, in a 2-row grid.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_modes(mesh.x, omega, modes, lambda p, x: np.sin(p * np.pi * x),
    ...                  mesh.conn[:, 0])  # doctest: +SKIP
    """
    xx = np.linspace(x[0], x[-1], 400)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, (n_modes + 1) // 2, figsize=(10, 5.5),
                                 sharex=True, sharey=True, constrained_layout=True)
        for ax, p in zip(axes.flat, range(1, n_modes + 1)):
            exact = exact_shape(p, xx)
            u = modes[:, p - 1].copy()
            # mode sign is arbitrary: align with the exact shape, then match amplitude
            u *= np.sign(np.dot(exact_shape(p, x), u)) / np.abs(u).max()
            ax.axhline(0, color=GRID, lw=1)
            ax.plot(xx, exact, color=INK_2, lw=1.5, ls="--", label="Exact")
            ax.plot(x, u, color=BLUE, label="FEM")
            ax.plot(x[element_nodes], u[element_nodes], color=BLUE, lw=0,
                    marker="o", ms=5, mec=SURFACE, mew=1, label="Element boundaries")
            ax.set_title(rf"$p={p}$,  $\omega={omega[p - 1]:.4f}$")
        for ax in axes[-1]:
            ax.set_xlabel("$x$")
        for ax in axes[:, 0]:
            ax.set_ylabel("Normalised displacement")
        axes.flat[0].legend(loc="lower right", fontsize=8)
    return fig


def animate(x, times, frames, ylim=(-1.1, 1.1), interval=30):
    """Animate displacement frames, e.g. from :func:`leapfrog`.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Node coordinates.
    times : ndarray, shape (n_frames,)
        Time of each frame, shown in the title.
    frames : ndarray, shape (n_frames, N)
        Displacement at each time.
    ylim : tuple of float, default (-1.1, 1.1)
        Fixed vertical limits.
    interval : int, default 30
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to it while it plays; save with ``.save("out.gif")``.

    Examples
    --------
    >>> anim = animate(mesh.x, times, frames)  # doctest: +SKIP
    >>> anim.save("string.gif", writer="pillow")  # doctest: +SKIP
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        (line,) = ax.plot(x, frames[0], color=BLUE)
        ax.set(xlim=(x[0], x[-1]), ylim=ylim, xlabel="$x$", ylabel="$u$")
        title = ax.set_title("")

    def update(i):
        line.set_ydata(frames[i])
        title.set_text(f"t = {times[i]:.2f}")
        return line, title

    return FuncAnimation(fig, update, frames=len(frames), interval=interval)


def _normalise_mode(u):
    """Scale to unit peak with a positive slope at the left end."""
    first = u[np.flatnonzero(np.abs(u) > 1e-12 * np.abs(u).max())[0]]
    return u * np.sign(first) / np.abs(u).max()


def plot_frequency_errors(names, rel_errs, tol=1e-3, title="Relative error against reference"):
    """Relative frequency error against mode number, one line per model.

    Parameters
    ----------
    names : sequence of str
        Model names for the legend; at most 4 (one per palette colour).
    rel_errs : sequence of ndarray
        Relative error of each mode, one array per model (lengths may differ).
        Zeros are drawn at 1e-16 so they appear on the log axis.
    tol : float, default 1e-3
        Drawn as a dotted reference line.
    title : str, default "Relative error against reference"
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If more than 4 models are given.

    Examples
    --------
    >>> fig = plot_frequency_errors(["uniform", "two-layer"], [err_u, err_2])  # doctest: +SKIP
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for name, err, color in zip(names, rel_errs, _series_colors(len(names))):
            p = np.arange(1, len(err) + 1)
            ax.semilogy(p, np.maximum(err, 1e-16), color=color, label=name,
                        marker="o", ms=4, mec=SURFACE, mew=1)
        ax.axhline(tol, color=INK_2, lw=1, ls=":")
        ax.text(1, tol * 1.3, f"tolerance {tol:g}", color=INK_2, fontsize=8)
        ax.set(xlabel="Mode number $p$", ylabel=r"$|\omega_p - \omega_p^{\rm ref}|\,/\,\omega_p^{\rm ref}$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.legend(loc="lower right")
        ax.set_title(title)
    return fig


def plot_models(names, xx, mus, rhos, xs, modes_list, omegas, n_modes=3):
    """One row per model: its mu and rho profiles, then its first mode shapes.

    Parameters
    ----------
    names : sequence of str
        Model names, one row each.
    xx : ndarray, shape (n_plot,)
        Points at which the profiles were sampled.
    mus, rhos : sequence of ndarray, each shape (n_plot,)
        Tension and density of each model on xx.
    xs : sequence of ndarray
        Node coordinates of each model's mesh.
    modes_list : sequence of ndarray, each shape (N_i, k_i)
        Mode shapes of each model, k_i >= n_modes.
    omegas : sequence of ndarray
        Frequencies of each model, shown in the legend.
    n_modes : int, default 3
        Modes per model; at most 4 (one per palette colour).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_models([d.model.name for d in discs], xx, mus, rhos,
    ...                   [d.mesh.x for d in discs], modes_list, omegas)  # doctest: +SKIP
    """
    mode_colors = _series_colors(n_modes)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(len(names), 2, figsize=(10, 2.4 * len(names)),
                                 sharex=True, constrained_layout=True, squeeze=False)
        for row, name in enumerate(names):
            ax_p, ax_m = axes[row]
            ax_p.plot(xx, mus[row], color=SERIES[0], label=r"$\mu$ (tension)")
            ax_p.plot(xx, rhos[row], color=SERIES[1], label=r"$\rho$ (density)")
            top = max(mus[row].max(), rhos[row].max())
            ax_p.set_ylim(0, 1.15 * top)
            ax_p.set_title(name)

            ax_m.axhline(0, color=GRID, lw=1)
            for p, color in zip(range(n_modes), mode_colors):
                ax_m.plot(xs[row], _normalise_mode(modes_list[row][:, p]), color=color,
                          label=rf"$p={p + 1}$, $\omega={omegas[row][p]:.3f}$")
            ax_m.set_ylim(-1.15, 1.15)
            ax_m.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            ax_m.set_title(f"{name}: first {n_modes} modes")
        axes[0, 0].legend(loc="lower right", fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("$x$")
    return fig
