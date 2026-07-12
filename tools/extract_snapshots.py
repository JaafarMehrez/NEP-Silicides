#!/usr/bin/env python3
"""
Extract snapshots from movie.xyz

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   July 2026
"""

import os
import numpy as np

base = "/path/to/md_sampling"
structures = ["C54_TiSi2", "C49_TiSi2", "TiSi_B27", "Ti5Si3"]
n_skip = 100 
n_target = 50


def parse_movie_xyz(filepath):
    frames = []
    with open(filepath) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        n_atoms = int(lines[i].strip())
        i += 1
        header = lines[i].strip()
        i += 1
        lat_str = header.split('"')[1]
        lat = np.array([float(x) for x in lat_str.split()]).reshape(3, 3)
        atoms = []
        for _ in range(n_atoms):
            parts = lines[i].strip().split()
            symbol = parts[0]
            pos = np.array([float(x) for x in parts[1:4]])
            atoms.append((symbol, pos))
            i += 1
        frames.append({"lattice": lat, "atoms": atoms})
    return frames


def write_poscar(frame, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    lat = frame["lattice"]
    atoms = frame["atoms"]
    
    symbols = []
    for s, _ in atoms:
        if s not in symbols:
            symbols.append(s)
    symbols.sort(key=lambda x: (x != "Ti", x))
    counts = [sum(1 for s, _ in atoms if s == sym) for sym in symbols]
    
    lat_inv = np.linalg.inv(lat)
    frac_pos = []
    for _, pos in atoms:
        frac = lat_inv @ pos
        frac = frac - np.floor(frac)
        frac_pos.append(frac)
        
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(f"{name} (NEP-MD snapshot)\n")
        f.write("1.0\n")
        for row in lat:
            f.write(f"  {row[0]:.15f}  {row[1]:.15f}  {row[2]:.15f}\n")
        f.write(" ".join(symbols) + "\n")
        f.write(" ".join(str(c) for c in counts) + "\n")
        f.write("Direct\n")
        for frac in frac_pos:
            f.write(f"  {frac[0]:.15f}  {frac[1]:.15f}  {frac[2]:.15f}\n")
            
for struct in structures:
    movie_path = os.path.join(base, struct, "movie.xyz")
    print(f"{struct}: reading {movie_path}")
    frames = parse_movie_xyz(movie_path)
    print(f"  {len(frames)} frames total")

    available = len(frames) - n_skip
    if available < n_target:
        print(f"  WARNING: only {available} frames after skipping, adjusting target")
        n = available
    else:
        n = n_target
        
    indices = np.linspace(n_skip, len(frames) - 1, n, dtype=int)
    
    snap_dir = os.path.join(base, struct, "snapshots")
    for i, idx in enumerate(indices):
        subdir = os.path.join(snap_dir, f"snapshot_{i + 1:04d}")
        write_poscar(frames[idx], subdir, "POSCAR")
        t_ps = idx * 0.001
        print(
            f"  snapshot {i + 1:3d}/{n}: frame {idx:4d} (t={t_ps:.3f} ns) -> {subdir}/POSCAR"
        )
        
    script_path = os.path.join(snap_dir, "convert_all.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Convert all POSCAR -> atom.config via poscar2config.x\n")
        f.write("CONVERTER=${1:-poscar2config.x}\n")
        f.write("for d in snapshot_*/; do\n")
        f.write('    if [ -f "${d}POSCAR" ]; then\n')
        f.write('        (cd "$d" && $CONVERTER POSCAR && echo "Converted $d")\n')
        f.write("    fi\n")
        f.write("done\n")
    os.chmod(script_path, 0o755)
    
print("\nDone. Snapshot directories and POSCAR files created.")
