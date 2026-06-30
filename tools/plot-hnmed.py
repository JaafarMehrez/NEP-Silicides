#!/usr/bin/env python3
"""
Plot Kappa results from HNMED calculation

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   June 2026
"""
import matplotlib
from matplotlib.pyplot import *
import numpy as np
from scipy.integrate import cumulative_trapezoid

matplotlib.rcParams.update(
    {
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
    }
)

labels_kappa = ["kxi", "kxo", "kyi", "kyo", "kz"]
run_dirs = ["run-00", "run-01", "run-02", "run-03"]


def running_ave(y, x):
    return cumulative_trapezoid(y, x, initial=0) / x


nsteps = 10000
t = np.arange(1, nsteps + 1) * 0.001

kappa_all = {key: np.zeros((len(run_dirs), nsteps)) for key in labels_kappa}
kappa_ra_all = {key: np.zeros((len(run_dirs), nsteps)) for key in labels_kappa}

for i, run_dir in enumerate(run_dirs):
    data = np.loadtxt(f"{run_dir}/kappa.out")
    for j, key in enumerate(labels_kappa):
        kappa_all[key][i, :] = data[:, j]
        kappa_ra_all[key][i, :] = running_ave(data[:, j], t)

total_ra = np.zeros((len(run_dirs), nsteps))
for i in range(len(run_dirs)):
    total_ra[i, :] = kappa_ra_all["kxi"][i, :] + kappa_ra_all["kxo"][i, :]

total_mean = np.mean(total_ra, axis=0)
total_std = np.std(total_ra, axis=0, ddof=1)

k_estimate = total_mean[-1]
k_err = total_std[-1]
print(f"Estimated thermal conductivity: {k_estimate:.2f} +/- {k_err:.2f} W/m/K")

fig = figure(figsize=(8, 7))
ax = gca()

for i in range(len(run_dirs)):
    plot(t, total_ra[i, :], "--", color="black", linewidth=0.8, alpha=0.6, label=None)

plot(t, total_mean, "-", color="blue", linewidth=2.5, label="Average")
fill_between(
    t,
    total_mean - total_std,
    total_mean + total_std,
    color="red",
    alpha=0.3,
    label=r"$\pm 1\sigma$",
)

xlim([0, 10])
ax.set_xticks(range(0, 11, 2))
ylim([-50, 200])
ax.set_yticks(range(-50, 201, 50))
xlabel("Time (ns)")
ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")

text_str = rf"$\kappa = {k_estimate:.1f} \pm {k_err:.1f}$ W m$^{{-1}}$ K$^{{-1}}$"
text(
    0.95,
    0.95,
    text_str,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=30,
    # bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

legend(loc="lower right", frameon=False)

tight_layout()
savefig("kappa_avg.png", dpi=300)
show()

