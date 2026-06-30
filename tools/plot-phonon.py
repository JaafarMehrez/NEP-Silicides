#!/usr/bin/env python3
"""
Plot phonon dispersion

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   June 2026
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save figure to file")
args = parser.parse_args()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "text.usetex": True,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.top": True,
    "ytick.right": True,
})


potentials = [
    ("mini-Tersoff", "mini-Tersoff/omega2.out", "black", "--", 1.5),
    ("NEP89", "nep89/omega2.out", "black", ":", 2.0),
    ("NEP89-FT", "nep89-Si-FT/omega2.out", "C4", "--", 1.5),
    ("NEP-expert", "nep-expert/omega2.out", "C2", "-.", 1.5),
    ("NEP-OMAT24", "nep-omat24/gen-00/omega2.out", "C0", "-", 2.0),
    ("DFT (PBEsol)", "phonondb_dft/omega2_dft.out", "red", "-", 1.5),
]

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which="major", length=tl, width=tw)
        ax.tick_params(which="minor", length=tlm, width=tw)
        ax.tick_params(which="both", axis="both", direction="in", right=True, top=True)

with open("nep-omat24/gen-00/omega2.out", "r") as f:
    first_line = f.readline().strip().lstrip("#").split()
    sym_points = first_line[: len(first_line) // 2]
    sym_points = [float(x) for x in sym_points]

plt.figure(figsize=(8, 8))
set_fig_properties([plt.gca()])

for label, path, color, linestyle, lw in potentials:
    omega2_array = np.loadtxt(path)
    linear_path = omega2_array[:, 0]
    nu = np.sqrt(np.abs(omega2_array[:, 1:])) / (2 * np.pi)
    nu[omega2_array[:, 1:] < 0] = np.nan
    lines = plt.plot(linear_path, nu, color=color, ls=linestyle, lw=lw)
    lines[0].set_label(label)

plt.xlim([0, sym_points[-1]])
plt.vlines(sym_points, ymin=0, ymax=17, color="gray", lw=1.5)
plt.gca().set_xticks(sym_points)
plt.gca().set_xticklabels([r"$\Gamma$", "X", "K", r"$\Gamma$", "L"])
plt.ylim([0, 17])
plt.ylabel(r"$\nu$ (THz)")
plt.legend(fontsize=18, frameon=False)
if args.save:
    plt.savefig("phonon_dispersion.png", dpi=300, bbox_inches="tight")
else:
    plt.show()

