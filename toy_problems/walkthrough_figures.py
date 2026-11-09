"""
Figures for the string walkthrough: the mesh and its GLL nodes, the reference
element, and the solution, adjoint and sensitivity kernel of

    -(mu u')' = 1 on [0, 1],  u(0) = u(1) = 0,  mu = 1 + x,

for J = u(0.5), each against its exact form. Also draws a radial mantle mesh
with a pressure field broken at the 660 km discontinuity, as the Stokes problem
will use.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fem_1d as fem
from fem_1d.plotting import BLUE, GRID, INK_2, MARKER, STYLE, plot_basis, plot_kernel, plot_mesh

OUT = Path(__file__).parent

# The string problem and its exact solution, adjoint and kernel
mu = lambda x: 1 + x
x0 = 0.5
L = lambda x: np.log1p(x)
u_exact = lambda x: L(x) / np.log(2) - x
du_exact = lambda x: (1 / np.log(2) - 1 - x) / (1 + x)
C1 = 1 - L(x0) / np.log(2)
lam_exact = lambda x: np.where(x < x0, C1 * L(x), (C1 - 1) * L(x) + L(x0))
dlam_exact = lambda x: np.where(x < x0, C1, C1 - 1) / mu(x)
K_exact = lambda x: -dlam_exact(x) * du_exact(x)

# Discretisation: 4 elements of degree 3, as in the walkthrough
n, Ne = 3, 4


def string_problem():
    space = fem.FunctionSpace.build(fem.uniform_mesh(1.0, Ne), n)
    return fem.Problem(
        space,
        terms=[fem.Term(lambda x: 1.0, d_test=1, d_trial=1, param="mu")],
        loads=[fem.LoadTerm(lambda x: 1.0)],
        constraints=[fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")],
        params={"mu": mu},
    )


def field_panel(ax, space, values, exact, title, ylabel):
    """A discrete field between its nodes, its nodal values, and the exact curve."""
    x = np.linspace(0, 1, 801)
    ax.axhline(0, color=GRID, lw=1)
    for xe in space.mesh.xe:
        ax.axvline(xe, color=GRID, lw=1, zorder=0)
    ax.plot(x, exact(x), color=INK_2, lw=1.5, ls="--", label="exact")
    ax.plot(x, space.evaluate(values, x), color=BLUE, label="FEM")
    ax.plot(space.x, values, lw=0, color=BLUE, label="GLL nodes", **MARKER)
    ax.set(title=title, ylabel=ylabel, xlabel="$x$")
    ax.legend(loc="lower center", fontsize=8)


def main():
    # 1. A radial mantle mesh: CMB, 660 and surface in planet radii
    mesh = fem.sized_mesh([0.546, 0.896, 1.0], drmax=0.06)
    fig = plot_mesh(
        {
            "U": fem.FunctionSpace.build(mesh, 6),
            "V": fem.FunctionSpace.build(mesh, 6),
            "P": fem.FunctionSpace.build(mesh, 4, discontinuous=True),
        },
        interfaces=[0.896],
        title="Mantle mesh, CMB to surface, 660 dashed",
    )
    fig.savefig(OUT / "mantle_mesh.png", dpi=200)

    # 2. The reference element
    plot_basis(fem.reference_element(4)).savefig(OUT / "reference_element.png", dpi=200)

    # 3. The string: mesh, solution, adjoint, kernel
    problem = string_problem()
    space = problem.spaces["u"]
    plot_mesh(space, title="String mesh").savefig(OUT / "string_mesh.png", dpi=200)

    sol = problem.solve()
    c = problem.functional("u", x0)
    lam = problem.adjoint(c)
    K = problem.kernel("mu", c=c, solution=sol)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        field_panel(axes[0], space, sol["u"], u_exact, "Forward: $A u = b$", "$u$")
        field_panel(axes[1], space, lam, lam_exact, rf"Adjoint: $A^T\lambda = c$, $J = u({x0})$", r"$\lambda$")
        plot_kernel(K, exact=K_exact, ax=axes[2], label=r"$-\lambda' u'$",
                    title=r"Kernel $K_\mu = -\lambda' u'$")
        axes[2].axvline(x0, color=INK_2, lw=1, ls=":")
    fig.savefig(OUT / "string_kernel.png", dpi=200)

    print(f"u(0.5) = {sol.evaluate('u', x0)[0]:.9f} (exact {u_exact(x0):.9f}); "
          f"kernel max error {np.abs(K.values - K_exact(K.x)).max():.1e}")


if __name__ == "__main__":
    main()
