#!/usr/bin/env python3
'''
Convert POSCAR to GPUMD extended XYZ format

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   July 2026
'''

import sys
from jarvis.core.atoms import Atoms

poscar = sys.argv[1]
out = (
    sys.argv[2]
    if len(sys.argv) > 2
    else poscar.replace(".vasp", ".xyz").replace("POSCAR_", "model_")
)

at = Atoms.from_poscar(poscar)
cart = at.cart_coords
elems = at.elements
lat = at.lattice_mat

with open(out, "w") as f:
    f.write(f"{at.num_atoms}\n")
    f.write(
        f'Lattice="{lat[0, 0]:.10f} {lat[0, 1]:.10f} {lat[0, 2]:.10f} '
        f"{lat[1, 0]:.10f} {lat[1, 1]:.10f} {lat[1, 2]:.10f} "
        f'{lat[2, 0]:.10f} {lat[2, 1]:.10f} {lat[2, 2]:.10f}" '
        f'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
    )
    for el, c in zip(elems, cart):
        f.write(f"    {el} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")

print(f"Written: {out} ({at.num_atoms} atoms)")
