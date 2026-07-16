#!/usr/bin/env python3
'''
Build Ti(001)/c-Si(001) interface with Ti strained to match Si.

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   July 2026
'''

import numpy as np
from jarvis.core.atoms import Atoms
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BULK = "/path/to/POSCAR_bulk"


A_SI = 5.43        # Si conventional cubic constant (Angstrom)
STRAIN_TI = 0.063  # Ti biaxial strain (6.3% expansion)
N_TI_BILAYERS = 4  # number of Ti bilayers (AB stacking)
N_SI_BILAYERS = 8  # number of Si bilayers along (001)
SEPARATION = 2.5   # Ti-Si interlayer spacing (Angstrom)
VACUUM = 15.0      # vacuum above Ti (Angstrom)


def make_si_slab(a, n_bilayers):
    atoms_frac = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
        ]
    )
    n_z = n_bilayers * 2
    cell = np.diag([a, a, a])
    coords = []
    for iz in range(n_z):
        z_frac = iz * 0.25
        cell_idx = iz // 4
        layer_idx = iz % 4
        for i in range(2):
            frac = atoms_frac[layer_idx * 2 + i].copy()
            frac[2] = z_frac
            cart = frac @ cell
            coords.append([cart[0], cart[1], cart[2]])
    return np.array(coords), cell


def make_ti_slab(a_xy, n_bilayers, strain):
    a_ti = 2.95
    c_ti = 4.68
    a_s = a_ti * (1.0 + strain)
    a1 = np.array([a_s, 0.0])
    a2 = np.array([-0.5 * a_s, 0.5 * np.sqrt(3) * a_s])
    layer_a = []
    for m in range(-10, 11):
        for n in range(-10, 11):
            p = m * a1 + n * a2
            if 0 <= p[0] < a_xy and 0 <= p[1] < a_xy:
                layer_a.append(p)
    layer_a = np.array(sorted(layer_a, key=lambda p: (p[1], p[0])))
    shift = (a1 + 2.0 * a2) / 3.0
    layer_b = layer_a + shift
    layer_b = np.mod(layer_b, [a_xy, a_xy])
    layer_b = np.array(sorted(layer_b, key=lambda p: (p[1], p[0])))
    z_ab = c_ti / 2.0
    coords = []
    for bi in range(n_bilayers):
        z0 = bi * c_ti
        for p in layer_a:
            coords.append([p[0], p[1], z0])
        for p in layer_b:
            coords.append([p[0], p[1], z0 + z_ab])
    return np.array(coords)

def main():
    si_coords, si_cell = make_si_slab(A_SI, N_SI_BILAYERS)
    z_si_max = si_coords[:, 2].max()
    z_si_min = si_coords[:, 2].min()
    n_si = len(si_coords)
    print(f"Si: {n_si} atoms, z range [{z_si_min:.2f}, {z_si_max:.2f}]")
    
    ti_coords = make_ti_slab(A_SI, N_TI_BILAYERS, STRAIN_TI)
    z_ti_min = ti_coords[:, 2].min()
    z_ti_max = ti_coords[:, 2].max()
    n_ti = len(ti_coords)
    print(f"Ti: {n_ti} atoms, z range [{z_ti_min:.2f}, {z_ti_max:.2f}]")
    
    from scipy.spatial import cKDTree
    
    tree = cKDTree(ti_coords[:, :2])
    dists, _ = tree.query(ti_coords[:, :2], k=2)
    nn = dists[:, 1].mean()
    print(f"Ti in-plane NN distance: {nn:.3f} Angstrom (bulk hcp: 2.95 Angstrom)")
    
    z_ti_offset = z_si_max + SEPARATION
    ti_coords[:, 2] += z_ti_offset - z_ti_min
    
    z_ti_min_new = ti_coords[:, 2].min()
    z_ti_max_new = ti_coords[:, 2].max()
    total_z = z_ti_max_new + VACUUM
    
    all_coords = np.vstack([ti_coords, si_coords])
    all_elems = ["Ti"] * n_ti + ["Si"] * n_si
    
    lattice = np.array([[A_SI, 0.0, 0.0], [0.0, A_SI, 0.0], [0.0, 0.0, total_z]])
    
    lat_inv = np.linalg.inv(lattice)
    all_frac = np.dot(all_coords, lat_inv)
    
    ti_mask = np.array([e == "Ti" for e in all_elems])
    si_mask = ~ti_mask
    
    ti_idx = np.where(ti_mask)[0]
    si_idx = np.where(si_mask)[0]
    ti_order = ti_idx[np.argsort(all_coords[ti_idx, 2])]
    si_order = si_idx[np.argsort(all_coords[si_idx, 2])]
    
    ordered_frac = np.vstack([all_frac[ti_order], all_frac[si_order]])
    ordered_elems = [all_elems[i] for i in list(ti_order) + list(si_order)]
    
    at = Atoms(
        lattice_mat=lattice,
        coords=ordered_frac,
        elements=ordered_elems,
        cartesian=False,
    )
    
    fname = "POSCAR_Ti(001)_Si(001).vasp"
    path = os.path.join(OUT_DIR, fname)
    at.write_poscar(path)
    print(f"\nWritten: {fname}")
    print(f"  {at.num_atoms} atoms ({n_ti} Ti + {n_si} Si)")
    print(f"  Cell: {A_SI:.2f} x {A_SI:.2f} x {total_z:.2f} Angstrom")
    print(f"  Ti z range: [{z_ti_min_new:.2f}, {z_ti_max_new:.2f}]")
    print(f"  Si z range: [{z_si_min:.2f}, {z_si_max:.2f}]")
    print(f"  Ti on top: {z_ti_min_new > z_si_max}")
    print(f"  Ti strain: {STRAIN_TI * 100:.1f}% (biaxial expansion)")

if __name__ == "__main__":
    main()
