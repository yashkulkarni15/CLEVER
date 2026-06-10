#!/bin/bash
#SBATCH --job-name=clever-p2-density
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-5%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p2_density_%A_%a.log
#SBATCH --error=logs/p2_density_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 2: Workload density sanity profiling.
# Array mapping:
#   0 lmsys theta=0.90   1 lmsys theta=0.30
#   2 qqp   theta=0.90   3 qqp   theta=0.30
#   4 moss  theta=0.90   5 moss  theta=0.30
#
# LMSYS still uses legacy flat paths. MOSS/QQP use Option-A nested paths.

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/density

DATASETS=(lmsys lmsys qqp qqp moss moss)
THETAS=(0.90 0.30 0.90 0.30 0.90 0.30)

DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
THETA="${THETAS[$SLURM_ARRAY_TASK_ID]}"
THETA_TAG="${THETA/./p}"
MODEL="all-MiniLM-L6-v2"
SIZE="100k"
OUT_DIR="results/density/phase2_${DS}_minilm_${SIZE}_theta_${THETA_TAG}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== [$(date)] Phase 2 density profile ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID:-local}"
echo "Node:     $(hostname)"
echo "Dataset:  $DS"
echo "Model:    $MODEL"
echo "Size:     $SIZE"
echo "Theta:    $THETA"
echo "Output:   $OUT_DIR"
echo ""

if [[ "$DS" == "lmsys" ]]; then
    python scripts/12_run_density_profile.py \
        --dataset "$DS" \
        --embedding-model "$MODEL" \
        --size "$SIZE" \
        --embeddings "results/embeddings/${SIZE}_embeddings.npy" \
        --queries "data/${SIZE}_queries.parquet" \
        --config configs/eviction.yaml \
        --output "$OUT_DIR" \
        --policies lru lfu \
        --cache-size 0.10 \
        --workload temporal \
        --density-theta "$THETA" \
        --checkpoints 10 \
        --probe-size 512 \
        --anchor-size 1024
else
    python scripts/12_run_density_profile.py \
        --dataset "$DS" \
        --embedding-model "$MODEL" \
        --size "$SIZE" \
        --config configs/eviction.yaml \
        --output "$OUT_DIR" \
        --policies lru lfu \
        --cache-size 0.10 \
        --workload temporal \
        --density-theta "$THETA" \
        --checkpoints 10 \
        --probe-size 512 \
        --anchor-size 1024
fi

echo ""
echo "=== [$(date)] Phase 2 density profile complete: $DS theta=$THETA ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
