# AUTHORSHIP: written by Claude (AI assistant) for Issac Jacob, Nov 2026.
"""Quick-look plot of the Earth kernels in kernels_*.npz:  python plot_kernels.py kernels_jump30.npz"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

k = np.load(sys.argv[1] if len(sys.argv) > 1 else "kernels_jump30.npz")
depth = (float(k["R"]) - k["r"]) / 1e3
o = np.argsort(depth)
names = {"h_b": r"surface topography $h_b$", "N": r"geoid $N$", "h_c": r"CMB topography $h_c$"}
fig, axs = plt.subplots(1, 3, figsize=(9, 4.2), sharey=True, constrained_layout=True)
cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
for ax, (n, title) in zip(axs, names.items()):
    for l, col in zip((2, 4, 8, 16, 32), cols):
        i = int(np.where(k["degrees"] == l)[0][0])
        ax.plot(k["K_" + n][i][o] * 1e3, depth[o], color=col, lw=1.1, label=rf"$\ell={l}$")
    ax.axhline(float(k["jump_depth"]) / 1e3, color="0.5", ls="--", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_title(title, fontsize=10); ax.set_xlabel(r"$K$ (m per kg m$^{-3}$ per km)")
axs[0].set_ylabel("depth (km)"); axs[0].invert_yaxis(); axs[0].legend(fontsize=8, frameon=False)
fig.suptitle(rf"Density kernels, viscosity $\times${float(k['jump']):g} below {float(k['jump_depth'])/1e3:.0f} km", fontsize=10)
fig.savefig(sys.argv[1].replace(".npz", ".png") if len(sys.argv) > 1 else "kernels.png", dpi=150)
