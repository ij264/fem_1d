"""
Waves on a fixed string: rho u_tt - (mu u_x)_x = 0 on [0, a], u(0) = u(a) = 0.

Spectral elements of degree n on GLL nodes (diagonal mass matrix). For each model
in MODELS, computes the normal modes, compares them with a fine high-order
reference solve, and propagates a Gaussian pulse from rest with leapfrog. The
uniform model is also checked against the exact omega_p = p pi c / a.

fem_1d assembles the stiffness and mass; the eigen solve and time stepping are
the few lines below.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.animation import PillowWriter
from scipy.linalg import eigh

import fem_1d as fem
from string_plots import animate, plot_frequencies, plot_frequency_errors, plot_models, plot_modes

OUT = Path(__file__).parent

ones = lambda x: np.ones_like(x)


@dataclass(frozen=True)
class Model:
    """A string of length a with tension mu(x), density rho(x) and jumps at interfaces."""

    name: str
    mu: Callable
    rho: Callable
    interfaces: tuple = ()
    a: float = 1.0


# Models: add or edit entries here. List any jumps in mu or rho as interfaces.
MODELS = [
    Model("uniform", mu=ones, rho=ones),
    Model("two-layer", mu=lambda x: np.where(x < 0.5, 1.0, 4.0), rho=ones, interfaces=(0.5,)),  # c = 1 then 2
    Model("tension gradient", mu=lambda x: 1.0 + 2.0 * x, rho=ones),
    Model("density bump", mu=ones, rho=lambda x: 1.0 + 2.0 * np.exp(-(((x - 0.6) / 0.1) ** 2))),
]

# Discretisation
n = 4  # polynomial degree (n + 1 nodes per element)
Ne = 10  # number of elements
n_ref, Ne_ref = 8, 4 * Ne  # reference solve: much finer and higher order

# Time stepping
T = 3.0  # total time
n_frames = 200


def discretise(model, n, Ne):
    """Space, stiffness K, lumped mass diagonal m and free-dof mask (both ends fixed)."""
    mesh = fem.sized_mesh([0.0, *model.interfaces, model.a], drmax=model.a / Ne)
    space = fem.FunctionSpace.build(mesh, n)
    K, _ = fem.assemble_system(space, [fem.Term(model.mu, 1, 1)])
    M, _ = fem.assemble_system(space, [fem.Term(model.rho)])  # diagonal on GLL nodes
    free = np.ones(space.n_dofs, dtype=bool)
    free[[0, -1]] = False
    return space, K, M.diagonal(), free


def normal_modes(K, m, free, k=None):
    """Frequencies and mass-normalised modes: K phi = w^2 diag(m) phi, via m^-1/2 K m^-1/2."""
    s = 1 / np.sqrt(m[free])
    B = K.toarray()[np.ix_(free, free)] * s[:, None] * s[None, :]
    k = int(free.sum()) if k is None else k
    w2, V = eigh(B, subset_by_index=(0, k - 1))
    modes = np.zeros((len(m), k))
    modes[free] = V * s[:, None]
    return np.sqrt(np.clip(w2, 0, None)), modes


def leapfrog(K, m, free, U0, dt, n_steps, save_every):
    """Central differences for m u_tt + K u = 0 from rest; returns saved times and frames."""
    Kf, mf = K[free][:, free], m[free]
    accel = lambda u: -(Kf @ u) / mf
    u = U0[free].copy()
    u_prev = u + 0.5 * dt**2 * accel(u)  # zero initial velocity
    times, frames = [0.0], [U0.copy()]
    for step in range(1, n_steps + 1):
        u, u_prev = 2 * u - u_prev + dt**2 * accel(u), u
        if step % save_every == 0:
            U = np.zeros_like(U0)
            U[free] = u
            times.append(step * dt)
            frames.append(U)
    return np.array(times), np.array(frames)


def solve(model):
    space, K, m, free = discretise(model, n, Ne)
    omega, modes = normal_modes(K, m, free)
    omega_ref, _ = normal_modes(*discretise(model, n_ref, Ne_ref)[1:], k=len(omega))

    U0 = np.exp(-(((space.x - 0.3) / 0.05) ** 2))
    U0[~free] = 0.0
    dt = 0.5 * 2 / omega[-1]  # half the leapfrog stability limit 2 / omega_max
    n_steps = int(T / dt)
    times, frames = leapfrog(K, m, free, U0, dt, n_steps, save_every=max(1, n_steps // n_frames))
    exact = (np.cos(np.outer(times[-1:], omega)) * (modes.T @ (m * U0))) @ modes.T  # modal superposition
    err = np.abs(frames[-1] - exact[0]).max() / np.abs(U0).max()
    print(f"{model.name:>17}: omega_1 = {omega[0]:.5f}, dt = {dt:.2e}, "
          f"leapfrog vs modal at t = {times[-1]:.2f}: {err:.1e}")
    return {
        "model": model, "space": space, "omega": omega, "modes": modes, "omega_ref": omega_ref,
        "times": times, "frames": frames,
    }


def slug(name):
    return name.replace(" ", "_").replace("-", "_")


def main():
    results = [solve(model) for model in MODELS]
    names = [r["model"].name for r in results]

    # Comparison across models.
    rel_errs = [np.abs(r["omega"] - r["omega_ref"]) / r["omega_ref"] for r in results]
    fig_err = plot_frequency_errors(
        names, rel_errs,
        title=f"Error against n={n_ref}, Ne={Ne_ref} reference (model: n={n}, Ne={Ne})",
    )
    xx = np.linspace(0.0, MODELS[0].a, 1001)
    fig_models = plot_models(
        names, xx,
        mus=[np.broadcast_to(m.mu(xx), xx.shape) for m in MODELS],
        rhos=[np.broadcast_to(m.rho(xx), xx.shape) for m in MODELS],
        xs=[r["space"].x for r in results],
        modes_list=[r["modes"] for r in results],
        omegas=[r["omega"] for r in results],
    )
    fig_err.savefig(OUT / "models_frequency_error.png", dpi=200)
    fig_models.savefig(OUT / "models.png", dpi=200)

    # Uniform string: exact frequencies and mode shapes are known.
    u = results[names.index("uniform")]
    space, a = u["space"], u["model"].a
    p = np.arange(1, len(u["omega"]) + 1)
    fig_f = plot_frequencies(
        u["omega"], p * np.pi / a,
        nodes_per_wavelength=lambda q: 2 * Ne * n / q,
        title=f"(Ne={Ne}, degree {n})",
    )
    fig_m = plot_modes(
        space.x, u["omega"], u["modes"],
        exact_shape=lambda q, x: np.sin(q * np.pi * x / a),
        element_nodes=space.boundary_dofs,
    )
    fig_f.savefig(OUT / "string_frequencies.png", dpi=200)
    fig_m.savefig(OUT / "string_modes.png", dpi=200)

    # One animation per model.
    ylim = max(np.abs(r["frames"]).max() for r in results) * 1.1
    for r in results:
        anim = animate(r["space"].x, r["times"], r["frames"], ylim=(-ylim, ylim))
        anim.save(OUT / f"string_{slug(r['model'].name)}.gif", writer=PillowWriter(fps=30))


if __name__ == "__main__":
    main()
