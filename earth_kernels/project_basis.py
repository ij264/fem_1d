# AUTHORSHIP: written by Claude (AI assistant) for Issac Jacob, Nov 2026.
"""Project the Earth kernels onto a radial basis: the forward operator of the inversion, degree by degree.

If the density anomaly at degree l is expanded in radial basis functions,
    drho_lm(r) = sum_j c_lm,j phi_j(r),
then each observable is
    d_lm = sum_j G_l,j c_lm,j,      G_l,j = int K_l(r) phi_j(r) dr,
with the same G_l for every order m. The integral uses the kernels' own quadrature points and
weights, so it is exact for any basis that is smooth within each finite element (the elements are
the 256 layers of the radial model, and the 666 km jump is one of their boundaries).

The measure matters only for adjoints and priors, not for G. G is defined against dr. If the
radial model space has the inner product <f, g> = int f g r^2 dr, the representer of the datum
d_lm is K_l(r) / r^2 there, not K_l(r). The Gram matrices in both measures are returned so either
convention can be used.

The basis provided here is piecewise linear in radius, continuous except at the viscosity jump,
where it has one coefficient on each side (density may jump there). Any other basis can be used
through ``project(kernels, phi)`` with phi(r) returning shape (n_basis, len(r)).

Usage:  python project_basis.py kernels_jump30.npz --layers-upper 12 --layers-lower 18 --out G_jump30.npz
"""
import argparse
import numpy as np


class HatBasis:
    """Piecewise-linear hat functions on knots, discontinuous at ``break_r`` (one dof on each side)."""

    def __init__(self, r_min, r_max, break_r, n_lower, n_upper):
        self.lower = np.linspace(r_min, break_r, n_lower + 1)
        self.upper = np.linspace(break_r, r_max, n_upper + 1)
        self.break_r = break_r
        self.knots = np.concatenate([self.lower, self.upper])  # the break appears twice
        self.n = self.knots.size

    def __call__(self, r):
        """Values phi_j(r), shape (n_basis,) + r.shape."""
        r = np.asarray(r, float)
        shape, r = r.shape, r.ravel()
        out = np.zeros((self.n, r.size))
        for off, kn, side in ((0, self.lower, r < self.break_r), (self.lower.size, self.upper, r >= self.break_r)):
            for j in range(kn.size):
                left = kn[j - 1] if j > 0 else kn[j]
                right = kn[j + 1] if j < kn.size - 1 else kn[j]
                up = (r >= left) & (r <= kn[j]) & side
                down = (r >= kn[j]) & (r <= right) & side
                with np.errstate(divide="ignore", invalid="ignore"):
                    if j > 0:
                        out[off + j, up] = (r[up] - left) / (kn[j] - left)
                    if j < kn.size - 1:
                        out[off + j, down] = (right - r[down]) / (right - kn[j])
                out[off + j, side & (r == kn[j])] = 1.0
        return out.reshape((self.n,) + shape)

    def coefficients(self, f):
        """Interpolation coefficients of f (values at the knots, each side of the break)."""
        eps = 1e-9 * (self.upper[-1] - self.lower[0])
        return np.concatenate([f(np.r_[self.lower[:-1], self.break_r - eps]), f(np.r_[self.break_r + eps, self.upper[1:]])])


def project(kernels, phi):
    """G[name] of shape (n_degrees, n_basis): G_l,j = sum_q w_q K_l(r_q) phi_j(r_q)."""
    r, w = kernels["r"], kernels["weights"]
    P = phi(r) * w  # (n_basis, n_q)
    return {name[2:]: kernels[name] @ P.T for name in kernels.files if name.startswith("K_")}


def gram(r, w, phi):
    """Gram matrices int phi_i phi_j dr and int phi_i phi_j r^2 dr, by the kernels' quadrature."""
    P = phi(r)
    return (P * w) @ P.T, (P * (w * r**2)) @ P.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernels")
    ap.add_argument("--layers-lower", type=int, default=18, help="hat intervals between the CMB and the jump")
    ap.add_argument("--layers-upper", type=int, default=12, help="hat intervals between the jump and the surface")
    ap.add_argument("--out", default="G.npz")
    a = ap.parse_args()
    k = np.load(a.kernels)
    basis = HatBasis(float(k["r_cmb"]), float(k["R"]), float(k["r_jump"]), a.layers_lower, a.layers_upper)
    G = project(k, basis)
    M_dr, M_r2dr = gram(k["r"], k["weights"], basis)
    np.savez(a.out, degrees=k["degrees"], knots=basis.knots, break_r=basis.break_r,
             M_dr=M_dr, M_r2dr=M_r2dr, **{f"G_{n}": g for n, g in G.items()},
             notes=np.array("d_l = G_l @ c_l, c_l the hat-basis coefficients of drho_lm(r) in kg/m^3, d in m; "
                            "knots in m, the break appears twice (one coefficient each side)"))
    print(f"{basis.n} basis functions; saved {a.out} with", ", ".join(f"G_{n} {g.shape}" for n, g in G.items()))


if __name__ == "__main__":
    main()
