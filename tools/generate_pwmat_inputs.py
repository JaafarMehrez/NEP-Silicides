#!/usr/bin/env python3
"""
Generate PWmat inputs for DFT snapshots

Author: Jaafar Mehrez
Email:  jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
Date:   July 2026
"""
import os
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
COMPOUNDS = ["C54_TiSi2", "C49_TiSi2", "TiSi_B27", "Ti5Si3"]

PSP_TI = os.environ.get("PSP_TI", "Ti.SG15.PBE.UPF")
PSP_SI = os.environ.get("PSP_SI", "Si.SG15.PBE.UPF")
ECUT = int(os.environ.get("PWMAT_ECUT", "50"))
ECUT2 = int(os.environ.get("PWMAT_ECUT2", "200"))
E_ERROR = os.environ.get("PWMAT_E_ERROR", "1.0e-6")
RHO_ERROR = os.environ.get("PWMAT_RHO_ERROR", "1.0e-6")

def make_etot_input(snapshot_dir, compound):
    path = os.path.join(snapshot_dir, "etot.input")
    if os.path.exists(path):
        print(f"  EXISTS: {path}")
        return
    content = f"""4  1
JOB = SCF
XCFUNCTIONAL = PBE
IN.PSP1 = {PSP_TI}
IN.PSP2 = {PSP_SI}
IN.ATOM = atom.config
ECUT = {ECUT}
ECUT2 = {ECUT2}
MP_N123 = 1 1 1 0 0 0
E_ERROR = {E_ERROR}
RHO_ERROR = {RHO_ERROR}
OUT.FORCE = T
OUT.STRESS = T
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"  CREATED: {path}")
    
def main():
    for compound in COMPOUNDS:
        compound_dir = os.path.join(BASE, compound, "snapshots")
        if not os.path.isdir(compound_dir):
            print(f"SKIP: {compound_dir} not found")
            continue
        snap_dirs = sorted(glob.glob(os.path.join(compound_dir, "snapshot_*")))
        print(f"\n{'=' * 60}")
        print(f"Compound: {compound}  —  {len(snap_dirs)} snapshots")
        print(f"{'=' * 60}")
        for sd in snap_dirs:
            if os.path.isdir(sd):
                label = os.path.basename(sd)
                make_etot_input(sd, compound)

if __name__ == "__main__":
    main()
