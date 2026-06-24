#!/usr/bin/env python3
"""
Plot NEP training results: loss curves and parity plots

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   June 2026
"""

import numpy as np
import matplotlib.pyplot as plt

DIR = "~/train"

loss = np.loadtxt(f"{DIR}/loss.out")
gen = loss[:, 0]
L_t = loss[:, 1]
L_1 = loss[:, 2]
L_2 = loss[:, 3]
e_train = loss[:, 4]  # eV/atom
f_train = loss[:, 5]  # eV/Angstrom
v_train = loss[:, 6]  # eV/atom
e_test = loss[:, 7]
f_test = loss[:, 8]
v_test = loss[:, 9]

e_train_mev = e_train * 1000
e_test_mev = e_test * 1000
v_train_mev = v_train * 1000
v_test_mev = v_test * 1000

print("=== Final RMSE (last generation) ===")
print(f"  Energy train : {e_train_mev[-1]:.2f} meV/atom")
print(f"  Energy test  : {e_test_mev[-1]:.2f}  meV/atom")
print(f"  Force train  : {f_train[-1]:.4f}  eV/Angstrom")
print(f"  Force test   : {f_test[-1]:.4f}   eV/Angstrom")
print(f"  Virial train : {v_train_mev[-1]:.2f} meV/atom")
print(f"  Virial test  : {v_test_mev[-1]:.2f}  meV/atom")

fig, ax = plt.subplots(figsize=(10, 6))

gen1000 = gen / 1000
ax.semilogy(gen1000, L_t, label="Total loss", lw=1)
ax.semilogy(gen1000, L_1, label="L1-regularization", lw=1, ls="--")
ax.semilogy(gen1000, L_2, label="L2-regularization", lw=1, ls=":")
ax.semilogy(
    gen1000, e_train, label=f"Energy-train (final={e_train_mev[-1]:.1f} meV/atom)", lw=1
)
ax.semilogy(gen1000, f_train, label=f"Force-train (final={f_train[-1]:.3f} eV/Angstrom)", lw=1)
ax.semilogy(
    gen1000, e_test, label=f"Energy-test (final={e_test_mev[-1]:.1f} meV/atom)", lw=1
)
ax.semilogy(gen1000, f_test, label=f"Force-test (final={f_test[-1]:.3f} eV/Angstrom)", lw=1)
ax.set_xlabel("Generation/1000")
ax.set_ylabel("Loss")
ax.set_title("NEP training loss")
ax.legend(ncol=1, fontsize=9)

plt.tight_layout()
plt.savefig(f"{DIR}/loss_curves.png", dpi=150)
print(f"\nSaved loss_curves.png")

e_train_data = np.loadtxt(f"{DIR}/energy_train.out")
e_train_pred = e_train_data[:, 0]
e_train_targ = e_train_data[:, 1]

e_test_data = np.loadtxt(f"{DIR}/energy_test.out")
e_test_pred = e_test_data[:, 0]
e_test_targ = e_test_data[:, 1]

f_train_data = np.loadtxt(f"{DIR}/force_train.out")
f_train_pred = f_train_data[:, :3].ravel()
f_train_targ = f_train_data[:, 3:].ravel()

f_test_data = np.loadtxt(f"{DIR}/force_test.out")
f_test_pred = f_test_data[:, :3].ravel()
f_test_targ = f_test_data[:, 3:].ravel()

v_train_data = np.loadtxt(f"{DIR}/virial_train.out")
v_train_pred = v_train_data[:, :6].ravel()
v_train_targ = v_train_data[:, 6:].ravel()

v_test_data = np.loadtxt(f"{DIR}/virial_test.out")
v_test_pred = v_test_data[:, :6].ravel()
v_test_targ = v_test_data[:, 6:].ravel()

def rmse(p, t):
    return np.sqrt(np.mean((p - t) ** 2))

def mae(p, t):
    return np.mean(np.abs(p - t))

def r2(p, t):
    cc = np.corrcoef(p, t)[0, 1]
    return cc**2

print("\n=== Parity metrics ===")
print(
    f"  Energy train: RMSE={rmse(e_train_pred, e_train_targ) * 1000:.1f} meV/atom, R2={r2(e_train_pred, e_train_targ):.4f}"
)
print(
    f"  Energy test:  RMSE={rmse(e_test_pred, e_test_targ) * 1000:.1f} meV/atom, R2={r2(e_test_pred, e_test_targ):.4f}"
)
print(
    f"  Force train:  RMSE={rmse(f_train_pred, f_train_targ):.4f} eV/Angstrom,  R2={r2(f_train_pred, f_train_targ):.4f}"
)
print(
    f"  Force test:   RMSE={rmse(f_test_pred, f_test_targ):.4f} eV/Angstrom,  R2={r2(f_test_pred, f_test_targ):.4f}"
)
print(
    f"  Virial train: RMSE={rmse(v_train_pred, v_train_targ) * 1000:.1f} meV/atom, R2={r2(v_train_pred, v_train_targ):.4f}"
)
print(
    f"  Virial test:  RMSE={rmse(v_test_pred, v_test_targ) * 1000:.1f} meV/atom, R2={r2(v_test_pred, v_test_targ):.4f}"
)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))


def parity_plot(ax, pred, targ, title, units, rmse_val, r2_val):
    lims = [min(pred.min(), targ.min()), max(pred.max(), targ.max())]
    pad = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.scatter(targ, pred, s=4, alpha=0.3, c="#1f77b4")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Target {units}")
    ax.set_ylabel(f"NEP {units}")
    ax.set_title(f"{title}\nRMSE={rmse_val:.3f}  R²={r2_val:.3f}")
    ax.set_aspect("equal")

parity_plot(
    axes[0, 0],
    e_train_pred,
    e_train_targ,
    "Energy (train)",
    "(eV/atom)",
    rmse(e_train_pred, e_train_targ),
    r2(e_train_pred, e_train_targ),
)
parity_plot(
    axes[0, 1],
    f_train_pred,
    f_train_targ,
    "Force (train)",
    "(eV/Angstrom)",
    rmse(f_train_pred, f_train_targ),
    r2(f_train_pred, f_train_targ),
)
parity_plot(
    axes[0, 2],
    v_train_pred,
    v_train_targ,
    "Virial (train)",
    "(eV/atom)",
    rmse(v_train_pred, v_train_targ),
    r2(v_train_pred, v_train_targ),
)
parity_plot(
    axes[1, 0],
    e_test_pred,
    e_test_targ,
    "Energy (test)",
    "(eV/atom)",
    rmse(e_test_pred, e_test_targ),
    r2(e_test_pred, e_test_targ),
)
parity_plot(
    axes[1, 1],
    f_test_pred,
    f_test_targ,
    "Force (test)",
    "(eV/Angstrom)",
    rmse(f_test_pred, f_test_targ),
    r2(f_test_pred, f_test_targ),
)
parity_plot(
    axes[1, 2],
    v_test_pred,
    v_test_targ,
    "Virial (test)",
    "(eV/atom)",
    rmse(v_test_pred, v_test_targ),
    r2(v_test_pred, v_test_targ),
)

plt.tight_layout()
plt.savefig(f"{DIR}/parity_plots.png", dpi=150)
print(f"Saved parity_plots.png")
