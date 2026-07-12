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
        cd "$SNAP" || exit
        mpirun -genvall -np $SLURM_NPROCS PWmat | tee output
        if [ $? -eq 0 ]; then
            echo "  FINISHED: $COMPOUND / $SNAP_NAME at $(date)"
        else
            echo "  FAILED: $COMPOUND / $SNAP_NAME at $(date)"
        fi
    done
done

echo ""
echo "=============================================="
echo "All calculations completed at $(date)"
echo "=============================================="
