#!/bin/bash
# Convert all POSCAR -> atom.config via poscar2config.x
CONVERTER=${1:-poscar2config.x}
for d in snapshot_*/; do
    if [ -f "${d}POSCAR" ]; then
        (cd "$d" && $CONVERTER POSCAR && echo "Converted $d")
    fi
done
