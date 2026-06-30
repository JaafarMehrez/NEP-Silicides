#!/usr/bin/env python3
"""
PWmat phonon dispersion workflow for hcp-Ti.

Steps:
  1. Run `PWmat` with etot.input (RELAX) to get relaxed structure
  2. Run this script to generate displaced supercells for phonopy
  3. Run PWmat SCF for each displacement
  4. Run phonopy to get phonon dispersion

Usage:
  python setup_phonon.py          # generate displacements
  python setup_phonon.py --parse  # parse forces and plot
"""

import numpy as np
import sys
import os
import subprocess

lattice = np.array(
    [
        [2.951, 0.0, 0.0],
        [-1.4755, 2.5556409665678785, 0.0],
        [0.0, 0.0, 4.679],
    ]
)

positions_frac = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.5],
    ]
)

atoms = ["Ti", "Ti"]
Z = 22

supercell = np.array([4, 4, 3])

def write_atom_config(filename, lat, pos_frac, atomic_numbers, frac=True):
    n = len(atomic_numbers)
    with open(filename, "w") as f:
        f.write(f"{n:5d}\n")
        f.write(" LATTICE\n")
        for row in lat:
            f.write(f"  {row[0]:20.10f} {row[1]:20.10f} {row[2]:20.10f}\n")
        f.write(" POSITION\n")
        for i in range(n):
            x, y, z = pos_frac[i]
            f.write(
                f"  {atomic_numbers[i]:3d}  {x:20.10f} {y:20.10f} {z:20.10f} 1 1 1\n"
            )

def write_etot_input(filename, atom_config, label=""):
    label_str = f"  # {label}" if label else ""
    with open(filename, "w") as f:
        f.write(f"4 1{label_str}\n")
        f.write("JOB = SCF\n")
        f.write("IN.PSP1 = Ti.SG15.PBE.UPF\n")
        f.write(f"IN.ATOM = {atom_config}\n")
        f.write("XCFUNCTIONAL = PBE\n")
        f.write("ECUT = 60\n")
        f.write("ECUT2 = 240\n")
        f.write("MP_N123 = 3 3 3 0 0 0\n")
        f.write("SMPLING_SIGMA = 0.02\n")
        f.write("OUT.FORCE = T\n")
        f.write("OUT.STRESS = F\n")


def generate_displacements():
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        print("phonopy not installed. Trying pip install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "phonopy"])
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        
    unitcell = PhonopyAtoms(
        symbols=atoms,
        cell=lattice,
        scaled_positions=positions_frac,
    )
    
    phonon = Phonopy(unitcell, supercell, is_symmetry=False)
    phonon.generate_displacements(distance=0.01)
    n_disp = len(phonon.supercells_with_displacements)
    print(
        f"Generated {n_disp} displaced supercells ({supercell[0]}x{supercell[1]}x{supercell[2]} supercell)"
    )
    os.makedirs("displacements", exist_ok=True)
    for i, sc in enumerate(phonon.supercells_with_displacements):
        if sc is None:
            continue
        pos_frac = sc.scaled_positions
        cell = sc.cell
        dirname = f"displacements/disp_{i + 1:03d}"
        os.makedirs(dirname, exist_ok=True)
        write_atom_config(f"{dirname}/atom.config", cell, pos_frac, [Z] * len(pos_frac))
        write_etot_input(
            f"{dirname}/etot.input", "atom.config", label=f"disp_{i + 1:03d}"
        )
        print(f"  Created {dirname}/")
    os.makedirs("displacements/disp_000", exist_ok=True)
    sc_perfect = phonon.supercell
    pos_frac = sc_perfect.scaled_positions
    cell = sc_perfect.cell
    write_atom_config(
        "displacements/disp_000/atom.config", cell, pos_frac, [Z] * len(pos_frac)
    )
    write_etot_input(
        "displacements/disp_000/etot.input", "atom.config", label="perfect"
    )
    print("  Created displacements/disp_000/ (perfect supercell)")
    
    import pickle
    with open("displacements/phonon.pkl", "wb") as f:
        pickle.dump(phonon, f)
    print(f"\nTotal: {n_disp} displacements + 1 perfect = {n_disp + 1} PWmat jobs")
    print("\nRun each PWmat job, e.g.:")
    print("  cd displacements/disp_001 && PWmat < etot.input > out.log")
    print("  # or submit all jobs via Slurm array")

def parse_forces():
    import pickle
    with open("displacements/phonon.pkl", "rb") as f:
        phonon = pickle.load(f)
    n_disp = len(phonon.supercells_with_displacements)
    natoms = len(phonon.supercell)
    forces = np.zeros((n_disp, natoms, 3))
    for i, sc in enumerate(phonon.supercells_with_displacements):
        if sc is None:
            continue
        dirname = f"displacements/disp_{i + 1:03d}"
        force_file = f"{dirname}/OUT.FORCE"
        if os.path.exists(force_file):
            f = open(force_file)
            f.readline()
            rows = []
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    int(parts[0])
                    rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
            f.close()
            rows = np.array(rows)
            if len(rows) > natoms:
                print(
                    f"  {dirname}/OUT.FORCE has {len(rows)} rows, expected {natoms}, taking last {natoms}"
                )
                rows = rows[-natoms:]
            elif len(rows) < natoms:
                print(
                    f"  WARNING: {dirname}/OUT.FORCE has only {len(rows)} rows, expected {natoms}"
                )
            forces[i] = rows
        else:
            print(f"  WARNING: No OUT.FORCE found in {dirname}/")
            found = os.listdir(dirname) if os.path.isdir(dirname) else []
            print(f"  Files in {dirname}/: {found}")
            
    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    
    path = [
        [0.0, 0.0, 0.0],  # G
        [1.0 / 3.0, 1.0 / 3.0, 0.0],  # K
        [0.5, 0.0, 0.0],  # M
        [0.0, 0.0, 0.0],  # G
        [0.0, 0.0, 0.5],  # A
    ]
    
    npoints = 401
    
    if hasattr(phonon, "get_band_structure"):
        bands = phonon.get_band_structure(path, npoints=npoints)
        distances = np.asarray(bands.distances)
        frequencies = bands.frequencies
    elif hasattr(phonon, "run_band_structure"):
        waypoints = [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
        n_per_seg = 101
        qpoints = []
        for i in range(len(waypoints) - 1):
            s = waypoints[i]
            e = waypoints[i + 1]
            n = n_per_seg + 1 if i == len(waypoints) - 2 else n_per_seg
            for j in range(n):
                t = j / n_per_seg
                qpoints.append([s[k] + t * (e[k] - s[k]) for k in range(3)])
        bs = phonon.run_band_structure([qpoints])
        seg_f = [np.asarray(f) for f in bs.frequencies]
        seg_f = [f.T if f.ndim == 2 and f.shape[0] > f.shape[1] else f for f in seg_f]
        frequencies = np.vstack(seg_f)
        sym_dists = [0.0, 1.418028, 2.127043, 3.356321, 4.027745]
        distances = []
        for i in range(len(waypoints) - 1):
            d0, d1 = sym_dists[i], sym_dists[i + 1]
            n = n_per_seg + 1 if i == len(waypoints) - 2 else n_per_seg
            for j in range(n):
                distances.append(d0 + j / n_per_seg * (d1 - d0))
        distances = np.array(distances)
    else:
        raise RuntimeError(
            "No compatible phonopy band structure API found. "
            "Try: pip install --upgrade phonopy"
        )
        
    frequencies = np.asarray(frequencies)
    if frequencies.ndim == 3:
        segs = [np.asarray(s) for s in frequencies]
        segs = [s.T if s.ndim == 2 and s.shape[0] > s.shape[1] else s for s in segs]
        frequencies = np.vstack(segs)
    if frequencies.shape[0] != len(distances):
        frequencies = frequencies.T
    nband = frequencies.shape[1]
    header = "# 0.0 1.418028 2.127043 3.356321 4.027745 G K M G A"
    omega2 = (2 * np.pi * frequencies) ** 2
    data_out = np.column_stack([distances, omega2])
    np.savetxt("omega2_dft.out", data_out, fmt="%.10f", header=header, comments="")
    print(f"Written omega2_dft.out ({len(data_out)} q-points, {nband} bands)")
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 22,
                "axes.linewidth": 2,
            }
        )
        plt.figure(figsize=(10, 10))
        for b in range(nband):
            plt.plot(distances, frequencies[:, b], "r-", lw=2.5)
        sym_points = [0.0, 1.418028, 2.127043, 3.356321, 4.027745]
        plt.xlim([0, sym_points[-1]])
        plt.vlines(sym_points, ymin=0, ymax=10, color="gray", lw=1.5)
        plt.xticks(sym_points, ["G", "K", "M", "G", "A"])
        plt.ylim([0, 10])
        plt.ylabel(r"$\nu$ (THz)")
        plt.savefig("phonon_dispersion_dft.png", dpi=300, bbox_inches="tight")
        print("Saved phonon_dispersion_dft.png")
    except ImportError:
        print("matplotlib not available, skipping plot")

if __name__ == "__main__":
    if "--parse" in sys.argv:
        parse_forces()
    else:
        generate_displacements()

