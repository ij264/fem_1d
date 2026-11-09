# AUTHORSHIP: written by Claude (AI assistant) for Issac Jacob, Nov 2026.
"""Cross-check the Earth kernels of earth_kernels.py against Sia's code on the same model.

Sia's code puts viscosity on its nodes and averages it over each layer, so a jump at node i is spread
over the layer below it. The finite-element model here puts the jump at that layer's midpoint.

Usage:  python check_sia.py --sia ~/Documents/dev/lyness [--fem path] [--jump 30]
"""
import argparse
import numpy as np

from earth_kernels import EarthModel

ap = argparse.ArgumentParser()
ap.add_argument("--sia", required=True)
ap.add_argument("--fem", default=None)
ap.add_argument("--jump", type=float, default=30.0)
ap.add_argument("--degrees", default="2,4,8,16,32")
ap.add_argument("--sia-layer", action="store_true", help="reproduce Sia's code's averaged layer at the jump")
a = ap.parse_args()
degrees = [int(s) for s in a.degrees.split(",")]

probe = EarthModel(a.sia, a.fem)
lm, fk, R = probe.sia_model, probe.fk, probe.R
i_j = int(np.argmin(np.abs(lm.r - (R - 670e3))))
eta_nodes = np.where(np.arange(lm.r.size) < i_j, a.jump, 1.0)
r_mid = 0.5 * (lm.r[i_j - 1] + lm.r[i_j])
# the layer of Sia's code straddling the jump has the average viscosity (midpoint rule); --sia-layer reproduces it exactly
r1, r2 = lm.r[i_j - 1] / R, lm.r[i_j] / R
eta_sia = lambda x: np.where(x < r1, a.jump, np.where(x < r2, 0.5 * (a.jump + 1.0), 1.0))
m = EarthModel(a.sia, a.fem, jump=a.jump, jump_depth=R - lm.r[i_j],
               eta=eta_sia if a.sia_layer else None, breaks=[r1, r2] if a.sia_layer else None)
ks = fk.kernels(lm.with_viscosity(eta_nodes), degrees)
print(f"viscosity x{a.jump:g} below {(R - r_mid) / 1e3:.1f} km depth (Sia's code node at {(R - lm.r[i_j]) / 1e3:.1f} km)")

away = np.abs(lm.r - r_mid) > 2 * np.max(np.diff(lm.r))  # skip the two layers either side of the jump
inner = slice(3, -3)
print(f"{'l':>3} {'observable':>10} {'rel-L2':>9} {'median ratio':>13} {'ratio range (away from jump)':>32}")
worst = 0.0
for j, l in enumerate(degrees):
    r, w, K = m.kernels(l)
    order = np.argsort(r)
    for name, ref in (("h_b", ks.surftopo[j]), ("N", ks.geoid[j]), ("h_c", ks.cmbtopo[j])):
        mine = np.interp(lm.r, r[order], K[name][order])
        sel = np.zeros(lm.r.size, bool); sel[inner] = True; sel &= away
        rel = np.linalg.norm(mine[sel] - ref[sel]) / np.linalg.norm(ref[sel])
        big = np.abs(ref) > 0.05 * np.abs(ref[sel]).max()
        ratio = mine[sel & big] / ref[sel & big]
        worst = max(worst, rel)
        print(f"{l:3d} {name:>10} {rel:9.2e} {np.median(ratio):13.5f}   [{ratio.min():.4f}, {ratio.max():.4f}]")
print(f"worst rel-L2 away from the jump: {worst:.2e}")
