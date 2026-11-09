# AUTHORSHIP: written by Claude (AI assistant) for Issac Jacob, Nov 2026.
"""Density kernels of surface topography, the geoid and CMB topography for the Earth model,
computed with fem_1d.self_gravity (compressible, self-gravitating, fluid core).

Model
-----
The radial reference model is the one from Sia's code (PREM densities on 257 nodes, its constants: water
1030 kg/m^3, core 9900 kg/m^3, core mass from GM_core). Viscosity is relative: eta = 1 above
the jump and ``jump`` below (30 by default) at ``jump_depth`` (670 km). Density kernels do not
depend on the absolute viscosity, only on its profile.

Scaling
-------
The problem is solved in non-dimensional units: r/R, rho/rho_s, g/g_s, eta/eta_upper, with
G_hat = G rho_s R / g_s (R surface radius, rho_s and g_s surface density and gravity). Tractions
scale with rho_s g_s R and the potential with g_s R. Every kernel returned is dimensional, with

    J = int_c^b K(r) drho(r) dr,     r in m, drho in kg/m^3, K in m / (kg/m^3) / m = m^3/kg / m,

so J comes out in metres for all four observables.

Observables (all positive upwards, metres)
------------------------------------------
    h_b    radial deflection of the surface:  h_b = -(T_b - rho_w Phi(b)) / ((rho(b) - rho_w) g(b)).
           With the ocean following the geoid, the surface mass is (rho(b) - rho_w) h_b + rho_w N = -T_b/g(b).
    N      geoid height (Bruns):              N = -Phi(b) / g(b),  physics-sign potential.
    h_b_N  seafloor height relative to the geoid, h_b - N: what a water depth measured from sea level sees.
    h_c    radial deflection of the CMB:      h_c = -(T_c - rho_core Phi(c)) / ((rho(c) - rho_core) g(c)).

T_b, T_c are the boundary tractions of fem_1d.self_gravity (stress positive in tension). The signs
of h_b, N and h_c reproduce the kernels of Sia's code (checked by check_sia.py). Which of h_b and
h_b_N the residual depths of Chapter 2 measure, and with which sign, is for Issac to confirm.

Usage
-----
    python earth_kernels.py --sia ~/Documents/dev/lyness --lmax 40 --out kernels_jump30.npz
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

OBSERVABLES = ("h_b", "N", "h_b_N", "h_c")


def import_packages(sia, fem_path=None):
    """Make flowkernels (Sia's code) and fem_1d importable; returns (fk, fem, sg, PointLoad, Problem)."""
    sys.path.insert(0, os.path.join(sia, "PYTHON"))
    if fem_path:
        sys.path.insert(0, fem_path)
    import flowkernels as fk
    import fem_1d as fem
    from fem_1d import self_gravity as sg
    from fem_1d.assembly import PointLoad
    from fem_1d.problem import Problem
    return fk, fem, sg, PointLoad, Problem


class EarthModel:
    """The non-dimensional Earth model and the scales that convert back to SI units."""

    def __init__(self, sia, fem_path=None, jump=30.0, jump_depth=670e3, drmax=0.01, n=5, eta=None, breaks=None, density="linear"):
        """eta (callable of r/R) and breaks (extra mesh radii, r/R) override the single jump, e.g. for a layered profile."""
        self.fk, self.fem, self.sg, self.PointLoad, self.Problem = import_packages(sia, fem_path)
        lm = self.fk.ShellModel.from_files(os.path.join(sia, "inputs.sph"),
                                           os.path.join(sia, "VISC_INPUTS", "const_visc.vis"))
        C = lm.constants
        self.sia_model, self.constants = lm, C
        self.R, self.r_cmb = float(lm.r[-1]), float(lm.r[0])
        self.rho_s, self.g_s = float(lm.rho[-1]), float(lm.gravity[-1])
        self.rho_w, self.rho_core = C.rho_water, C.rho_core
        self.G_hat = C.G * self.rho_s * self.R / self.g_s
        self.jump, self.jump_depth = jump, jump_depth

        x = lm.r / self.R
        self.nodes = x
        if density == "linear":
            # the model of Sia's code as it stands: density linear in each layer between its 257 nodes, so PREM's
            # discontinuities are layers of steep gradient, exactly as Sia's code solves them; one element per layer
            vals, slope = lm.rho / self.rho_s, np.diff(lm.rho / self.rho_s) / np.diff(x)
            seg = lambda r: np.clip(np.searchsorted(x, r) - 1, 0, slope.size - 1)
            self.rho0 = lambda r: np.interp(r, x, vals)
            self.kappa = lambda r: r * slope[seg(r)] / np.interp(r, x, vals)
        else:
            spl = CubicSpline(x, lm.rho / self.rho_s)
            self.rho0 = lambda r: spl(r)
            self.kappa = lambda r: r * spl(r, 1) / spl(r)
        rho = self.rho0
        self.b, self.c = 1.0, self.r_cmb / self.R
        rho_core_mean = C.gm_core / (C.G * (4 * np.pi / 3) * self.r_cmb**3)
        self.g0 = self.sg.gravity_profile(self.rho0, self.c, self.b, self.G_hat, rho_core=rho_core_mean / self.rho_s)
        self.r_jump = (self.R - jump_depth) / self.R
        if density == "linear" and eta is None:  # put the jump on the nearest model node (666 km for 670)
            self.r_jump = float(x[np.argmin(np.abs(x - self.r_jump))])
        self.jump_depth_used = self.R * (1 - self.r_jump)
        self.eta = eta or (lambda r: np.where(r < self.r_jump, jump, 1.0))

        fem = self.fem
        knots = {self.c, self.b, *(breaks if breaks is not None else [self.r_jump])}
        if density == "linear":
            knots |= set(x)
        knots = sorted(knots)
        mesh = fem.sized_mesh(knots, drmax=drmax)
        vel = fem.FunctionSpace(mesh, fem.reference_element(n))
        pres = fem.FunctionSpace(mesh, fem.reference_element(n - 2), discontinuous=True)
        self.spaces = self.sg.spaces(vel, pres)
        self.kw = dict(rho_fluid_c=self.rho_core / self.rho_s)  # ocean follows the geoid: rho_fluid_b = 0

        # dimensional conversion of each observable: J = factor * J_hat, with the adjoint loads of J_hat
        gb, gc = float(self.g0(self.b)) * self.g_s, float(self.g0(self.c)) * self.g_s
        rho_c_mantle = float(rho(self.c)) * self.rho_s
        rw, rc = self.rho_w / self.rho_s, self.rho_core / self.rho_s
        self.g_b, self.g_c = gb, gc
        self.observables = {
            # name: (j_b, j_c, jphi_b, jphi_c, factor for kernels per unit drho_hat dr_hat -> SI)
            "h_b": (1.0, 0.0, -rw, 0.0, -self.g_s / ((self.rho_s - self.rho_w) * gb)),
            "N": (0.0, 0.0, 1.0, 0.0, -self.g_s / (self.rho_s * gb)),
            "h_c": (0.0, 1.0, 0.0, -rc, -self.g_s / ((rho_c_mantle - self.rho_core) * gc)),
        }

    def params(self, drho=None):
        return {"eta": self.eta, "drho": drho or (lambda r: np.zeros_like(r))}

    def forward(self, l, drho=None):
        return self.sg.forward_problem(self.spaces, l, self.rho0, self.kappa, self.g0, self.G_hat,
                                       self.b, self.c, self.params(drho), **self.kw)

    def adjoint(self, l, j_b, j_c, jphi_b, jphi_c):
        terms = self.sg.adjoint_terms(l, self.rho0, self.kappa, self.g0, self.G_hat, self.b, self.c, **self.kw)
        loads = self.sg.observable_loads(j_b, j_c, jphi_b)
        if jphi_c:
            loads.append(self.PointLoad(jphi_c, "left", test="Phi"))
        return self.Problem(self.spaces, terms, loads, (), self.params()).solve()

    def kernels(self, l):
        """Dimensional kernels at degree l: r (m), weights (m), and K[name] on the quadrature points."""
        fwd = self.forward(l)
        out = {}
        for name, (jb, jc, jpb, jpc, factor) in self.observables.items():
            K = fwd.kernel("drho", adjoint=self.adjoint(l, jb, jc, jpb, jpc))
            # J = R rho_s factor J_hat and J_hat = int K_hat drho_hat dr_hat = (1/(rho_s R)) int K_hat drho dr,
            # so J = int (factor K_hat) drho dr
            out[name] = factor * K.values.ravel()
        out["h_b_N"] = out["h_b"] - out["N"]
        r = K.x.ravel() * self.R
        w = K.weights.ravel() * self.R
        return r, w, out

    def observables_direct(self, l, drho_si):
        """Observables from a forward solve, for a density anomaly drho_si(r) in kg/m^3 (r in m)."""
        sol = self.forward(l, drho=lambda x: drho_si(x * self.R) / self.rho_s).solve()
        T_b, T_c = self.sg.tractions(sol)
        phi_b = float(self.sg.potential(sol, self.b)[0])
        phi_c = float(self.sg.potential(sol, self.c)[0])
        vals = {}
        for name, (jb, jc, jpb, jpc, factor) in self.observables.items():
            J_hat = jb * T_b + jc * T_c + jpb * phi_b + jpc * phi_c
            vals[name] = self.R * self.rho_s * factor * J_hat  # T, rho Phi scale with rho_s g_s R
        vals["h_b_N"] = vals["h_b"] - vals["N"]
        return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sia", required=True, help="Folder of Sia's code (inputs.sph, VISC_INPUTS, PYTHON)")
    ap.add_argument("--fem", default=None, help="folder containing the fem_1d package, if not installed")
    ap.add_argument("--lmin", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=40)
    ap.add_argument("--jump", type=float, default=30.0)
    ap.add_argument("--jump-depth", type=float, default=670e3)
    ap.add_argument("--drmax", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="kernels.npz")
    a = ap.parse_args()

    m = EarthModel(a.sia, a.fem, a.jump, a.jump_depth, a.drmax, a.n)
    degrees = np.arange(a.lmin, a.lmax + 1)
    K = {name: [] for name in OBSERVABLES}
    t = time.time()
    for l in degrees:
        r, w, k = m.kernels(int(l))
        for name in OBSERVABLES:
            K[name].append(k[name])
        print(f"l = {l:3d} done ({time.time() - t:.0f} s)", flush=True)
    print(f"viscosity x{a.jump:g} below {m.jump_depth_used / 1e3:.1f} km depth")
    np.savez(a.out, degrees=degrees, r=r, weights=w, **{f"K_{n}": np.array(K[n]) for n in OBSERVABLES},
             R=m.R, r_cmb=m.r_cmb, r_jump=m.r_jump * m.R, jump_depth=m.jump_depth_used, jump=a.jump,
             notes=np.array("J_l = sum(weights * K_l * drho_l(r)) [m]; drho in kg/m^3; r, weights in m; "
                            "rows of K_* are degrees; see earth_kernels.py for sign conventions"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
