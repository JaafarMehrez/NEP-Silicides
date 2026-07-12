#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --job-name=DFT-Bulk-TiSilicide
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1

module load mkl mpi
module load cuda/12.4
module load pwmat/20260420

export I_MPI_FABRICS=ofi

BASE_DIR=$(pwd)
COMPOUNDS="C49_TiSi2 C54_TiSi2 Ti5Si3 TiSi_B27"
FAILED_LOG="$BASE_DIR/failed_snapshots.txt"
: > "$FAILED_LOG"

for COMPOUND in $COMPOUNDS; do
    echo ""
    echo "=============================================="
    echo "Starting compound: $COMPOUND at $(date)"
    echo "=============================================="
    SNAP_PARENT="$BASE_DIR/$COMPOUND/snapshots"
    for SNAP in "$SNAP_PARENT"/snapshot_*; do
        [ -d "$SNAP" ] || continue
        SNAP_NAME=$(basename "$SNAP")
        echo ""
        echo "  Running: $COMPOUND / $SNAP_NAME at $(date)"
        cd "$SNAP" || { echo "  FAILED to cd into $SNAP"; continue; }

        mpirun -genvall -np $SLURM_NPROCS PWmat > output 2>&1; MPIRUN_EXIT=$?
        if [ $MPIRUN_EXIT -eq 0 ]; then
            echo "  FINISHED: $COMPOUND / $SNAP_NAME at $(date)"
        else
            echo "  FAILED (exit=$MPIRUN_EXIT): $COMPOUND / $SNAP_NAME at $(date)"
            echo "$COMPOUND/$SNAP_NAME" >> "$FAILED_LOG"
        fi

        sleep 5
    done
done

echo ""
echo "=============================================="
echo "All calculations completed at $(date)"
echo "=============================================="
if [ -s "$FAILED_LOG" ]; then
    echo "The following snapshots FAILED:"
    cat "$FAILED_LOG"
fi
