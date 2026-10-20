"""
We solve the equation \\rho \\partial_t^2 u - \\partial_x (\\mu \\partial_x u) = f, where \\rho is the density and \\mu is the tension.
The solution is computed using the finite element method with linear elements.
f is a source term that is applied to the system.
The solution is computed over a time interval [0, T] with time step dt.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre
from tqdm import tqdm


def derivative_matrix_gll(xi, n):
    P = Legendre.basis(n)(xi)
    diff = xi[:, None] - xi[None, :]
    np.fill_diagonal(diff, 1.0)
    D = (P[:, None] / P[None, :]) / diff
    np.fill_diagonal(D, 0.0)
    D[0, 0] = -n * (n + 1) / 4
    D[-1, -1] = n * (n + 1) / 4
    return D


# Reference element: GLL points, weights, derivative matrix
n = 4  # polynomial degree (n + 1 nodes per element)
Pn = Legendre.basis(n)
xi = np.concatenate(([-1.0], np.sort(Pn.deriv().roots().real), [1.0]))
w = 2.0 / (n * (n + 1) * Pn(xi) ** 2)
D = derivative_matrix_gll(xi, n)

# Mesh
a, Ne = 1.0, 10
xe = np.linspace(0, a, Ne + 1)  # element end points
he = np.diff(xe)  # element lengths
x = np.zeros(Ne * n + 1)
for e in range(Ne):
    idx = np.arange(e * n, e * n + n + 1)
    x[idx] = xe[e] + 0.5 * (1 + xi) * he[e]

mu = lambda x: np.ones_like(x)
# Stiffness matrix K.
K = np.zeros((Ne * n + 1, Ne * n + 1))
for e in range(Ne):
    idx = np.arange(e * n, e * n + n + 1)
    mu_q = mu(x[idx])  # mu at this element's nodes
    Ke = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        for l in range(n + 1):
            for q in range(n + 1):
                Ke[k, l] += w[q] * mu_q[q] * D[q, k] * D[q, l]
    Ke *= 2 / he[e]
    K[np.ix_(idx, idx)] += Ke

# Now construct the mass matrix M.
rho = lambda x: np.ones_like(x)  # density function
M = np.zeros((Ne * n + 1, Ne * n + 1))
for e in range(Ne):
    idx = np.arange(e * n, e * n + n + 1)
    rho_q = rho(x[idx])
    Me = np.diag(w * rho_q) * he[e] / 2
    M[np.ix_(idx, idx)] += Me

# Apply boundary conditions.
K_ff = K[1:-1, 1:-1]
M_ff = M[1:-1, 1:-1]
x_free = x[1:-1]

from scipy.linalg import eigh

w2, modes = eigh(K_ff, M_ff)
omega = np.sqrt(w2)

# Now we solve the time-dependent problem using the eigenmodes.
omega_max = omega[-1]
dt = 0.5 * 2 / omega_max  # half the stability limit 2/omega_max
T = 3.0  # total time (c = 1, a = 1: 3 transits of the string)
n_steps = int(T / dt)

# Initial wave.
U0 = np.exp(-(((x_free - 0.3) / 0.05) ** 2))

# 3. time loop
A = np.linalg.solve(M_ff, K_ff)
U_prev = U0 - 0.5 * dt**2 * (A @ U0)
U_now = U0.copy()

save_every = max(1, n_steps // 200)  # keep about 200 frames
frames, times = [U0], [0.0]
for m in tqdm(range(1, n_steps + 1)):
    U_next = 2 * U_now - U_prev - dt**2 * (A @ U_now)
    U_prev, U_now = U_now, U_next
    if m % save_every == 0:
        frames.append(U_now.copy())
        times.append(m * dt)
frames = np.array(frames)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots()
(line,) = ax.plot(x_free, frames[0])
ax.set_xlim(0, a)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")
title = ax.set_title("")


def update(i):
    line.set_ydata(frames[i])
    title.set_text(f"t = {times[i]:.2f}")
    return line, title


anim = FuncAnimation(fig, update, frames=len(frames), interval=30)
anim.save("string.gif", writer=PillowWriter(fps=30))  # writes a file
plt.show()  # or view it live
# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)

c = 1.0  # wave speed sqrt(mu / rho) for the uniform string
p = np.arange(1, len(omega) + 1)
omega_exact = p * np.pi * c / a

# Figure 1: eigenfrequencies against the analytic p*pi*c/a.
fig1, (ax_f, ax_e) = plt.subplots(
    1, 2, figsize=(10, 4), constrained_layout=True
)

ax_f.plot(
    p, omega_exact, color=INK_2, lw=1.5, ls="--", label=r"Exact $p\pi c/a$"
)
ax_f.plot(
    p,
    omega,
    color=BLUE,
    lw=0,
    marker="o",
    ms=6,
    mec="#fcfcfb",
    mew=1.5,
    label="FEM",
)
ax_f.set(xlabel="Mode number $p$", ylabel=r"Angular frequency $\omega_p$")
ax_f.legend(loc="upper left")

rel_err = np.abs(omega - omega_exact) / omega_exact
ax_e.semilogy(p, rel_err, color=BLUE, marker="o", ms=6, mec="#fcfcfb", mew=1.5)
ax_e.set(
    xlabel="Mode number $p$", ylabel=r"$|\omega_p - p\pi c/a|\,/\,(p\pi c/a)$"
)
ax_e.grid(True, which="major")
# Resolved range: modes before the error first exceeds tol.
tol = 1e-3
p_res = p[np.argmax(rel_err > tol)] - 1 if (rel_err > tol).any() else p[-1]
ax_e.axhline(tol, color=INK_2, lw=1, ls=":")
ax_e.text(p[0], tol * 1.3, f"tolerance {tol:g}", color=INK_2, fontsize=8)

# Nodes per wavelength, 2 Ne n / p, on a secondary axis.
nodes_per_wavelength = lambda q: 2 * Ne * n / np.maximum(q, 1e-9)
for ax in (ax_f, ax_e):
    ax.axvspan(p_res + 0.5, p[-1] + 0.5, color=ORANGE, alpha=0.10, lw=0)
    ax.set_xlim(0.5, p[-1] + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    top = ax.secondary_xaxis(
        "top", functions=(nodes_per_wavelength, nodes_per_wavelength)
    )
    top.set_xlabel("Nodes per wavelength", fontsize=9)
    top.set_xticks(
        [
            t
            for t in (20, 10, 8, 6, 5, 4, 3)
            if nodes_per_wavelength(p[0]) >= t >= nodes_per_wavelength(p[-1])
        ]
    )
    top.spines["top"].set_visible(False)
    top.xaxis.set_tick_params(length=0)
ax_f.text(
    p_res + 1,
    ax_f.get_ylim()[0],
    "unresolved",
    color=ORANGE,
    fontsize=9,
    va="bottom",
    fontweight="bold",
)
ax_f.set_title("Eigenfrequencies of the fixed string", pad=34)
ax_e.set_title(f"Relative error (Ne={Ne}, degree {n})", pad=34)

# Figure 2: first four mode shapes (small multiples, shared axes).
x_full = x
fig2, axes = plt.subplots(
    2, 2, figsize=(10, 5.5), sharex=True, sharey=True, constrained_layout=True
)
xx = np.linspace(0, a, 400)
for ax, pi_ in zip(axes.flat, range(1, 5)):
    u = np.concatenate(([0.0], modes[:, pi_ - 1], [0.0]))
    exact = np.sin(pi_ * np.pi * xx / a)
    # mode sign is arbitrary: align with the analytic shape, then match amplitude
    u *= np.sign(np.dot(np.interp(x_full, xx, exact), u)) / np.abs(u).max()
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(xx, exact, color=INK_2, lw=1.5, ls="--", label="Exact")
    ax.plot(x_full, u, color=BLUE, lw=2, label="FEM")
    ax.plot(
        x_full[::n],
        u[::n],
        color=BLUE,
        lw=0,
        marker="o",
        ms=5,
        mec="#fcfcfb",
        mew=1,
        label="Element boundaries",
    )
    ax.set_title(rf"$p={pi_}$,  $\omega={omega[pi_ - 1]:.4f}$")
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
for ax in axes[1]:
    ax.set_xlabel("$x$")
for ax in axes[:, 0]:
    ax.set_ylabel("Normalised displacement")
axes[0, 0].legend(loc="lower right", fontsize=8)

if __name__ == "__main__":
    fig1.savefig("string_frequencies.png", dpi=200)
    fig2.savefig("string_modes.png", dpi=200)
