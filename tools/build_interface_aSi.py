#!/usr/bin/env python3
'''
Build Ti/amorphous-Si interface with Ti-commensurate cell.

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   July 2026
'''

import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
A_SI_XYZ = os.path.join(OUT_DIR, "216_R1.xyz") # find the structure in /data folder

A_TI = 2.95
C_TI = 4.68
A2_Y = 0.5 * np.sqrt(3) * A_TI
AB_SHIFT = np.array([(A_TI + 2.0 * (-0.5 * A_TI)) / 3.0, 2.0 * A2_Y / 3.0])

SEPARATION = 2.5
VACUUM = 15.0
N_TI_BILAYERS = 3

M = 6
N = 6

def read_xyz(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    cell_str = lines[1].split('"')[1]
    cell = np.array([float(x) for x in cell_str.split()]).reshape(3, 3)
    coords = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[2 + i].strip().split()
        coords[i] = [float(x) for x in parts[1:4]]
    return coords, cell

def wrap_coords(coords, cell):
    frac = coords @ np.linalg.inv(cell)
    frac = frac - np.floor(frac)
    return frac @ cell

def strain_cell(coords, old_cell, new_cell):
    frac = coords @ np.linalg.inv(old_cell)
    return frac @ new_cell

def make_ti_slab(lx, ly, n_bilayers):
    a1 = np.array([A_TI, 0.0])
    a2 = np.array([-0.5 * A_TI, A2_Y])
    
    layer_a = []
    for n in range(N):
        for m in range(-10, 11):
            p = m * a1 + n * a2
            if 0 <= p[0] < lx and 0 <= p[1] < ly:
                layer_a.append(p)
    layer_a = np.array(sorted(layer_a, key=lambda p: (p[1], p[0])))
    layer_b = layer_a + AB_SHIFT
    
    z_ab = C_TI / 2.0
    coords = []
    for bi in range(n_bilayers):
        z0 = bi * C_TI
        for p in layer_a:
            coords.append([p[0], p[1], z0])
        for p in layer_b:
            coords.append([p[0], p[1], z0 + z_ab])
    return np.array(coords), len(layer_a), len(layer_b)

def write_poscar(coords, elements, lattice, out_path):
    n = len(coords)
    symbols = []
    for el in elements:
        if el not in symbols:
            symbols.append(el)
    symbols.sort(key=lambda x: (x != "Ti", x))
    counts = [sum(1 for el in elements if el == sym) for sym in symbols]
    
    lat_inv = np.linalg.inv(lattice)
    idx_by_sym = []
    for sym in symbols:
        idx_by_sym.extend([i for i, el in enumerate(elements) if el == sym])
        
    frac = coords[idx_by_sym] @ lat_inv
    frac = frac - np.floor(frac)
    
    with open(out_path, "w") as f:
        f.write("Ti/a-Si interface (Ti-commensurate cell)\n")
        f.write("1.0\n")
        for row in lattice:
            f.write(f"  {row[0]:.15f}  {row[1]:.15f}  {row[2]:.15f}\n")
        f.write(" ".join(symbols) + "\n")
        f.write(" ".join(str(c) for c in counts) + "\n")
        f.write("Direct\n")
        for fv in frac:
            f.write(f"  {fv[0]:.15f}  {fv[1]:.15f}  {fv[2]:.15f}\n")

def main():
    si_coords, si_cell = read_xyz(A_SI_XYZ)
    print(
        f"a-Si: {len(si_coords)} atoms, cell {si_cell[0, 0]:.3f} x {si_cell[1, 1]:.3f} x {si_cell[2, 2]:.3f}"
    )
    
    si_coords = wrap_coords(si_coords, si_cell)
    
    lx = M * A_TI
    ly = N * A2_Y
    lz = si_cell[2, 2]
    new_cell = np.diag([lx, ly, lz])
    si_coords = strain_cell(si_coords, si_cell, new_cell)
    
    z_min = si_coords[:, 2].min()
    z_max = si_coords[:, 2].max()
    print(f"  z range (strained): [{z_min:.3f}, {z_max:.3f}]")
    
    ti_coords, n_a, n_b = make_ti_slab(lx, ly, N_TI_BILAYERS)
    n_ti = len(ti_coords)
    print(
        f"Ti: {n_ti} atoms ({N_TI_BILAYERS} bilayers, {n_a} per A-layer, {n_b} per B-layer)"
    )
    
    z_ti_min = ti_coords[:, 2].min()
    z_ti_max = ti_coords[:, 2].max()
    print(f"  z range: [{z_ti_min:.3f}, {z_ti_max:.3f}]")
    print(f"  in-plane NN: {A_TI:.3f}, AB shift: {AB_SHIFT[1]:.3f}")
    
    z_ti_offset = z_max + SEPARATION
    ti_coords[:, 2] += z_ti_offset - z_ti_min
    z_ti_min_new = ti_coords[:, 2].min()
    z_ti_max_new = ti_coords[:, 2].max()
    total_z = z_ti_max_new + VACUUM
    
    all_coords = np.vstack([ti_coords, si_coords])
    all_elems = ["Ti"] * n_ti + ["Si"] * len(si_coords)
    lattice = np.diag([lx, ly, total_z])
    
    fname = "POSCAR_Ti_a-Si.vasp"
    out_path = os.path.join(OUT_DIR, fname)
    write_poscar(all_coords, all_elems, lattice, out_path)
    
    n_si = len(si_coords)
    print(f"\nWritten: {fname}")
    print(f"  {len(all_coords)} atoms ({n_ti} Ti + {n_si} Si)")
    print(f"  Cell: {lx:.3f} x {ly:.3f} x {total_z:.3f}")
    print(f"  a-Si z range: [{z_min:.3f}, {z_max:.3f}]")
    print(f"  Ti z range: [{z_ti_min_new:.3f}, {z_ti_max_new:.3f}]")
    print(f"  Ti on top: {z_ti_min_new > z_max}")
    print(f"  Ti-commensurate: m={M}, n={N} -> Lx={lx:.3f}, Ly={ly:.3f}")

if __name__ == "__main__":
    main()
