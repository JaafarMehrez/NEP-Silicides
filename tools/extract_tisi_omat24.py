#!/usr/bin/env python3
"""
Extract Ti/Si-only structures from an OMat24 sub-dataset directory.
Usage: python extract_tisi_omat24.py <input_dir> <output_xyz>
Example: python extract_tisi_omat24.py ./train/rattled-relax ./omat24_ti_si_only.xyz

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   June 2026
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from ase.io import write
from ase.db import connect

TI_Z = 22
SI_Z = 14
ALLOWED_Z = {TI_Z, SI_Z}


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_tisi_omat24.py <input_dir> <output_xyz>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)

    aselmdb_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.aselmdb"), recursive=True)
    )
    if not aselmdb_files:
        print(f"ERROR: No .aselmdb files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(aselmdb_files)} aselmdb files in {os.path.basename(input_dir)}")

    all_structures = []
    for db_file in aselmdb_files:
        basename = os.path.basename(db_file)
        try:
            db = connect(db_file)
            n_total = db.count()
            for row in tqdm(db.select(), total=n_total, desc=f"  {basename[:30]:30s}"):
                atoms = row.toatoms()
                atomic_nums = set(atoms.get_atomic_numbers())

                if atomic_nums.issubset(ALLOWED_Z) and len(atomic_nums) > 0:
                    all_structures.append(atoms)

        except Exception as e:
            print(f"  Skipping {basename}: {e}")
            continue

    if not all_structures:
        print("\nNo pure Ti/Si structures found!")
        sys.exit(0)

    print(f"\nWriting {len(all_structures)} Ti/Si-only structures to {output_file}")
    write(output_file, all_structures, format="extxyz")

    from collections import Counter

    formulas = [a.get_chemical_formula() for a in all_structures]
    print(f"\nTop compositions:")
    for formula, count in Counter(formulas).most_common(20):
        pct = count / len(all_structures) * 100
        print(f"  {formula:20s}: {count:6d} ({pct:5.2f}%)")

    n_ti_only = sum(1 for a in all_structures if set(a.get_atomic_numbers()) == {TI_Z})
    n_si_only = sum(1 for a in all_structures if set(a.get_atomic_numbers()) == {SI_Z})
    n_tisi = sum(
        1 for a in all_structures if set(a.get_atomic_numbers()) == {TI_Z, SI_Z}
    )
    print(f"\nBreakdown:")
    print(f"  Pure Ti:  {n_ti_only}")
    print(f"  Pure Si:  {n_si_only}")
    print(f"  Ti-Si:    {n_tisi}")
    print(f"  Total:    {len(all_structures)}")


if __name__ == "__main__":
    main()
