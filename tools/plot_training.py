#!/usr/bin/env python3

"""
Plot NEP training results: loss curves and parity plots

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   June 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "text.usetex": True, 
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
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

DIR = "~/path/to/your/training/folder"

loss = np.loadtxt(f"{DIR}/loss.out")
gen = loss[:, 0]
L_t = loss[:, 1]
L_1 = loss[:, 2]
L_2 = loss[:, 3]
e_train = loss[:, 4]
f_train = loss[:, 5]

fig, ax = plt.subplots(figsize=(6, 6))


colors = {
    'total': '#004c8c', 
    'l1': '#2ca02c',    
    'l2': '#ff8c00',    
    'energy': '#e41a1c',
    'force': '#6a3d9a'  
}

ax.loglog(gen, L_t, label="Total", color=colors['total'], lw=1.5)
ax.loglog(gen, L_1, label="L1", color=colors['l1'], lw=1.5)
ax.loglog(gen, L_2, label="L2", color=colors['l2'], lw=1.5)
ax.loglog(gen, e_train, label="Energy", color=colors['energy'], lw=1.5)
ax.loglog(gen, f_train, label="Force", color=colors['force'], lw=1.5)


ax.set_xlabel("Generation")
ax.set_ylabel("Loss")
ax.set_xlim(gen[1], gen[-1])
ax.set_ylim(1e-3, 1)

ax.legend(frameon=False, loc='lower left', handlelength=0.8, borderpad=0.1, labelspacing=0.2)

plt.tight_layout()
plt.savefig(f"{DIR}/loss_curves.png", dpi=300, bbox_inches='tight')
print("Saved loss_curves.png")

e_train_data = np.loadtxt(f"{DIR}/energy_train.out")
e_test_data = np.loadtxt(f"{DIR}/energy_test.out")
f_train_data = np.loadtxt(f"{DIR}/force_train.out")
f_test_data = np.loadtxt(f"{DIR}/force_test.out")

def rmse(p, t): return np.sqrt(np.mean((p - t) ** 2))

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

def styled_parity(ax, pred, targ, label, units, color):
    low = min(pred.min(), targ.min())
    high = max(pred.max(), targ.max())
    ax.plot([low, high], [low, high], 'k--', lw=1, zorder=1)
    ax.scatter(targ, pred, s=10, alpha=0.4, color=color, edgecolors='none', zorder=2)
    
    ax.set_title(f"{label} (RMSE: {rmse(pred, targ):.3f} {units})", fontsize=14)
    ax.set_xlabel(f"Target ({units})")
    ax.set_ylabel(f"NEP ({units})")
    ax.set_aspect('equal')

styled_parity(axes[0,0], e_train_data[:,0], e_train_data[:,1], "Energy Train", "eV/at", "#1f77b4")
styled_parity(axes[0,1], e_test_data[:,0], e_test_data[:,1], "Energy Test", "eV/at", "#ff7f0e")

styled_parity(axes[1,0], f_train_data[:,:3].ravel(), f_train_data[:,3:].ravel(), "Force Train", "eV/Å", "#2ca02c")
styled_parity(axes[1,1], f_test_data[:,:3].ravel(), f_test_data[:,3:].ravel(), "Force Test", "eV/Å", "#d62728")

plt.tight_layout()
plt.savefig(f"{DIR}/parity_plots.png", dpi=300)
print("Saved parity_plots.png")

