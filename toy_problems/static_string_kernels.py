"""
Static sag of a string under gravity, and its adjoint sensitivity kernels.

Forward problem:  -(mu u')' = rho g  on [0, a],  u(0) = u(a) = 0.
Measurement:      J = u(x_r), the sag at the receiver point x_r.
Adjoint problem:  -(mu lam')' = delta(x - x_r),  lam(0) = lam(a) = 0.
Kernels:          dJ = int [ K_mu dmu + K_rho drho ] dx  with
                  K_mu = -lam' u',   K_rho = g lam.

Derivation: the discrete forward problem is K u = f. Perturbing it,
dK u + K du = df, so du = K^-1 (df - dK u). For J = e_r . u (e_r picks out
node r), dJ = e_r . du = lam . (df - dK u) with K lam = e_r (K is symmetric).
lam . dK u is int dmu lam' u' dx and lam . df is int drho g lam dx.

Discretised with spectral elements of degree n on GLL nodes, as in
waves_on_string_basic.py.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre


def derivative_matrix_gll(xi, n):
    P = Legendre.basis(n)(xi)
    diff = xi[:, None] - xi[None, :]
    np.fill_diagonal(diff, 1.0)
    D = (P[:, None] / P[None, :]) / diff
    np.fill_diagonal(D, 0.0)
    D[0, 0] = -n * (n + 1) / 4
    D[-1, -1] = n * (n + 1) / 4
    return D


# Problem
a = 1.0  # length of the string
g = 9.81  # gravitational acceleration (u > 0 is downward sag)
mu = lambda x: np.ones_like(x)  # tension
rho = lambda x: np.ones_like(x)  # density
x_r = 0.3  # receiver: where we measure the sag

# Reference element: GLL points, weights, derivative matrix
n = 4  # polynomial degree (n + 1 nodes per element)
Pn = Legendre.basis(n)
xi = np.concatenate(([-1.0], np.sort(Pn.deriv().roots().real), [1.0]))
w = 2.0 / (n * (n + 1) * Pn(xi) ** 2)
D = derivative_matrix_gll(xi, n)

# Mesh
Ne = 10
xe = np.linspace(0, a, Ne + 1)  # element end points
he = np.diff(xe)  # element lengths
x = np.zeros(Ne * n + 1)
for e in range(Ne):
    idx = np.arange(e * n, e * n + n + 1)
    x[idx] = xe[e] + 0.5 * (1 + xi) * he[e]

# Stiffness matrix K and diagonal mass M (as before).
K = np.zeros((Ne * n + 1, Ne * n + 1))
M = np.zeros((Ne * n + 1, Ne * n + 1))
for e in range(Ne):
    idx = np.arange(e * n, e * n + n + 1)
    mu_q = mu(x[idx])
    rho_q = rho(x[idx])
    Ke = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        for l in range(n + 1):
            for q in range(n + 1):
                Ke[k, l] += w[q] * mu_q[q] * D[q, k] * D[q, l]
    Ke *= 2 / he[e]
    K[np.ix_(idx, idx)] += Ke
    M[np.ix_(idx, idx)] += np.diag(w * rho_q) * he[e] / 2

# Load vector: f_k = int rho g phi_k dx = g * M_kk (GLL quadrature).
f = g * np.diag(M)

# Apply boundary conditions: drop the two fixed end nodes.
K_ff = K[1:-1, 1:-1]
f_ff = f[1:-1]

# Forward solve.
u = np.zeros(Ne * n + 1)
u[1:-1] = np.linalg.solve(K_ff, f_ff)

# Measurement: x_r must be a node (here an element boundary).
r = np.argmin(np.abs(x - x_r))
assert np.isclose(x[r], x_r), "put x_r on a node"
J = u[r]

# Adjoint solve: same K, with a unit point load at the receiver.
e_r = np.zeros(Ne * n + 1)
e_r[r] = 1.0
lam = np.zeros(Ne * n + 1)
lam[1:-1] = np.linalg.solve(K_ff, e_r[1:-1])

# Kernels at each element's GLL nodes, kept per element (u' jumps between elements).
K_mu = np.zeros((Ne, n + 1))
K_rho = np.zeros((Ne, n + 1))
x_q = np.zeros((Ne, n + 1))
for e in range(Ne):
    idx = np.arange(e * n, e * n + n + 1)
    du_e = (2 / he[e]) * D @ u[idx]  # u'(x) at the nodes
    dlam_e = (2 / he[e]) * D @ lam[idx]  # lam'(x) at the nodes
    K_mu[e] = -dlam_e * du_e
    K_rho[e] = g * lam[idx]
    x_q[e] = x[idx]

# Checks.
u_exact = g * x * (a - x) / 2  # uniform string: mu = rho = 1
print(f"J = u(x_r) = {J:.6f}, exact {g * x_r * (a - x_r) / 2:.6f}")
print(f"max |u - u_exact| = {np.abs(u - u_exact).max():.1e}")


def sag_at_receiver(dmu, drho):
    """Re-solve with mu + dmu and rho + drho; return u(x_r)."""
    K = np.zeros((Ne * n + 1, Ne * n + 1))
    f = np.zeros(Ne * n + 1)
    for e in range(Ne):
        idx = np.arange(e * n, e * n + n + 1)
        mu_q = mu(x[idx]) + dmu(x[idx])
        rho_q = rho(x[idx]) + drho(x[idx])
        Ke = np.einsum("q,q,qk,ql->kl", w, mu_q, D, D) * 2 / he[e]
        K[np.ix_(idx, idx)] += Ke
        f[idx] += g * w * rho_q * he[e] / 2
    u = np.zeros(Ne * n + 1)
    u[1:-1] = np.linalg.solve(K[1:-1, 1:-1], f[1:-1])
    return u[r]


# Finite-difference check: perturb with a small bump, compare with int K dm dx.
bump = lambda x: 0.01 * np.exp(-(((x - 0.6) / 0.1) ** 2))
zero = lambda x: np.zeros_like(x)
dx_q = w * (he / 2)[:, None]  # quadrature weights for int ... dx
for name, kernel, args in [
    ("mu", K_mu, ((bump, zero), (lambda x: -bump(x), zero))),
    ("rho", K_rho, ((zero, bump), (zero, lambda x: -bump(x)))),
]:
    predicted = np.sum(dx_q * kernel * bump(x_q))
    fd = (sag_at_receiver(*args[0]) - sag_at_receiver(*args[1])) / 2
    print(f"dJ from {name:>3} bump: kernel {predicted:.6e}, finite difference {fd:.6e}")

# Plots.
fig, (ax_u, ax_k) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
ax_u.plot(x, u, label=r"forward $u$ (sag)")
ax_u.plot(x, lam * g, label=r"adjoint $g\lambda$")
ax_u.axvline(x_r, color="0.6", ls=":", label=r"receiver $x_r$")
ax_u.set(ylabel="displacement", title="Forward and adjoint fields")
ax_u.legend()
for e in range(Ne):
    ax_k.plot(x_q[e], K_mu[e], color="C0", label=r"$K_\mu$" if e == 0 else None)
    ax_k.plot(x_q[e], K_rho[e], color="C1", label=r"$K_\rho$" if e == 0 else None)
ax_k.axvline(x_r, color="0.6", ls=":")
ax_k.axhline(0, color="0.8", lw=1)
ax_k.set(xlabel="$x$", ylabel="kernel", title=r"Sensitivity of $u(x_r)$")
ax_k.legend()

if __name__ == "__main__":
    fig.savefig("static_kernels.png", dpi=200)
