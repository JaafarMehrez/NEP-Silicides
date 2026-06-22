#!/usr/bin/env python3
"""
Compute NEP-like radial descriptors, PCA

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   June 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ase.io import read
from ase.neighborlist import neighbor_list
from sklearn.decomposition import PCA
from collections import Counter

print("Loading structures...")
atoms_list = read(
    "~/train.xyz",
    index=":",
    format="extxyz",
)
print(f"Loaded {len(atoms_list)} structures")

RC = 8.0  # radial cutoff (Å)
N_BASIS = 8  # number of Chebyshev basis functions

def cutoff_fn(r, rc):
    return np.where(r < rc, 0.5 * (1.0 + np.cos(np.pi * r / rc)), 0.0)

def chebyshev_basis(r, rc, n_basis):
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    x = np.clip(x, -1.0, 1.0)
    basis = np.zeros((len(r), n_basis))
    basis[:, 0] = 1.0
    if n_basis > 1:
        basis[:, 1] = x
    for k in range(2, n_basis):
        basis[:, k] = 2.0 * x * basis[:, k - 1] - basis[:, k - 2]
    return basis


def structure_descriptor(atoms, rc=RC, n_basis=N_BASIS, n_bins=30):
    i_arr, j_arr, d_arr, shift_arr = neighbor_list(
        "ijdS", atoms, cutoff=rc, self_interaction=False
    )
    
    if len(d_arr) == 0:
        return np.zeros(n_basis + n_bins)
    
    fc = cutoff_fn(d_arr, rc)
    basis = chebyshev_basis(d_arr, rc, n_basis)
    radial_desc = np.sum(fc[:, None] * basis, axis=0)
    hist, _ = np.histogram(d_arr, bins=n_bins, range=(1.0, rc), weights=fc)
    
    return np.concatenate([radial_desc, hist])

print("Computing NEP-like radial descriptors...")
descriptors = []
compositions = []
n_atoms_list = []

for idx, a in enumerate(atoms_list):
    if idx % 500 == 0:
        print(f"  {idx}/{len(atoms_list)}")
        
    desc = structure_descriptor(a)
    descriptors.append(desc)
    
    nums = a.get_atomic_numbers()
    n_ti = sum(1 for z in nums if z == 22)
    n_si = sum(1 for z in nums if z == 14)
    n_atoms_list.append(len(nums))
    
    if n_ti > 0 and n_si > 0:
        compositions.append("Ti-Si")
    elif n_ti > 0:
        compositions.append("Ti")
    else:
        compositions.append("Si")
        
descriptors = np.array(descriptors)
print(f"Descriptor matrix shape: {descriptors.shape}")
print("Running PCA...")

desc_mean = descriptors.mean(axis=0)
desc_std = descriptors.std(axis=0)
desc_std[desc_std == 0] = 1.0
desc_norm = (descriptors - desc_mean) / desc_std

pca = PCA(n_components=2)
coords = pca.fit_transform(desc_norm)
print(
    f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}"
)
print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")

print("Plotting...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = {"Si":"#1f77b4", "Ti": "#ff7f0e", "Ti-Si": "#2ca02c"}
markers = {"Si": "o", "Ti": "s", "Ti-Si": "^"}
ax = axes[0]
for comp in ["Si", "Ti", "Ti-Si"]:
    mask = np.array([c == comp for c in compositions])
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        c=colors[comp],
        marker=markers[comp],
        label=f"{comp} ({mask.sum()})",
        alpha=0.6,
        s=15,
        edgecolors="none",
    )
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("(a) By composition")
ax.legend(markerscale=2, fontsize=9)
ax = axes[1]
sc = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=n_atoms_list,
    cmap="viridis",
    alpha=0.6,
    s=15,
    edgecolors="none",
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("(b) By number of atoms")
plt.colorbar(sc, ax=ax, label="N atoms")

ax = axes[2]
energies = np.array([a.get_potential_energy() / len(a) for a in atoms_list])
sc = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=energies,
    cmap="coolwarm",
    alpha=0.6,
    s=15,
    edgecolors="none",
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("(c) By energy/atom (eV)")
plt.colorbar(sc, ax=ax, label="Energy/atom (eV)")
plt.tight_layout()
plt.savefig(
    "~/pca_descriptor_space.png",
    dpi=150,
)
print("Saved to pca_descriptor_space.png")
plt.close()
