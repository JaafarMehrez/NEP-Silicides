#!/bin/bash
# Batch process OMat24 sub-datasets
# Usage:
#   bash batch_process.sh            # process all train sub-datasets
#   bash batch_process.sh train      # process all train sub-datasets
#   bash batch_process.sh val        # process all val sub-datasets

set -euo pipefail

SPLIT="${1:-train}"
BASE="~/NEP/nep-silicide/data/OMat24"
SCRIPT="$BASE/extract_tisi_omat24.py"
OUTDIR="$BASE/$SPLIT"
TARDIR="$BASE/tar"
mkdir -p "$TARDIR" "$OUTDIR"

if [ "$SPLIT" = "train" ]; then
    BASE_URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train"
elif [ "$SPLIT" = "val" ]; then
    BASE_URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val"
else
    echo "Usage: $0 [train|val]"
    exit 1
fi

SUBDATASETS=(
    "rattled-1000"
    "rattled-1000-subsampled"
    "rattled-500"
    "rattled-500-subsampled"
    "rattled-300"
    "rattled-300-subsampled"
    "aimd-from-PBE-1000-npt"
    "aimd-from-PBE-1000-nvt"
    "aimd-from-PBE-3000-npt"
    "aimd-from-PBE-3000-nvt"
    "rattled-relax"
)

for sub in "${SUBDATASETS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing: $SPLIT/$sub"
    echo "======================================================================"

    TAR_FILE="$TARDIR/${sub}.tar.gz"
    SUBDIR="$OUTDIR/$sub"
    XYZ_FILE="$BASE/omat24_${SPLIT}_${sub}_ti_si_only.xyz"

    if [ -f "$XYZ_FILE" ] && [ -s "$XYZ_FILE" ]; then
        if [[ "$(tail -c 1 "$XYZ_FILE" 2>/dev/null)" == "" ]]; then
            n_lines=$(wc -l < "$XYZ_FILE")
            if [ "$n_lines" -ge 4 ]; then
                echo "  [SKIP]  $XYZ_FILE already exists (~$((n_lines / 2)) structures)"
                continue
            fi
        fi
        echo "  [FIX]   Removing partial XYZ file from interrupted run..."
        rm -f "$XYZ_FILE"
    fi

    if [ -f "$TAR_FILE" ]; then
        echo "  [RESUME] $sub.tar.gz (existing partial file, resuming...)"
    else
        echo "  [DOWNLOAD] $sub.tar.gz"
    fi
    curl -C - -f -L -o "$TAR_FILE" "$BASE_URL/${sub}.tar.gz"
    echo "  [DONE]  Download complete"

    echo "  [CHECK] Verifying tar.gz integrity..."
    if ! gzip -t "$TAR_FILE" 2>/dev/null; then
        echo "  [ERROR] $TAR_FILE is still corrupt. Delete it manually and re-run."
        exit 1
    fi
    echo "  [OK]    tar.gz is intact"

    n_aselmdb=$(find "$SUBDIR" -name "*.aselmdb" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_aselmdb" -eq 0 ]; then
        echo "  [EXTRACT] $sub.tar.gz -> $SUBDIR"
        mkdir -p "$SUBDIR"
        tar -xzf "$TAR_FILE" --strip-components=1 -C "$SUBDIR"
        echo "  [DONE]  Extraction complete"
        n_aselmdb=$(find "$SUBDIR" -name "*.aselmdb" | wc -l | tr -d ' ')
        echo "  [INFO]  Extracted $n_aselmdb aselmdb files"
    else
        echo "  [SKIP]  $n_aselmdb aselmdb files already present in $SUBDIR"
    fi

    echo "  [EXTRACT] Ti/Si structures..."
    python3 "$SCRIPT" "$SUBDIR" "$XYZ_FILE"

    if [ -f "$XYZ_FILE" ] && [ -s "$XYZ_FILE" ]; then
        echo "  [OK]    $(wc -l < "$XYZ_FILE") lines written to $(basename "$XYZ_FILE")"
    else
        echo "  [WARN]  XYZ file is empty or missing"
    fi

    echo "  [CLEANUP] Removing raw aselmdb files..."
    find "$SUBDIR" -type f \( -name "*.aselmdb" -o -name "*.aselmdb-lock" -o -name "*.npz" \) -delete 2>/dev/null || true
    find "$SUBDIR" -type d -empty -delete 2>/dev/null || true
    rm -f "$TAR_FILE"
    echo "  [OK]    Cleanup complete"

    df -h "$BASE" | awk 'NR==2{print "  [DISK]  " $4 " free on " $NF}'
done

echo ""
echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All $SPLIT sub-datasets processed!"
echo "======================================================================"
echo ""
echo "Next step: merge all XYZ files for $SPLIT:"
echo "  head -n 999999999 $(ls $BASE/omat24_${SPLIT}_*_ti_si_only.xyz | tr '\n' ' ') > omat24_${SPLIT}_ti_si_merged.xyz"
echo "  wc -l $BASE/omat24_${SPLIT}_ti_si_merged.xyz"
