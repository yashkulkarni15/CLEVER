#!/bin/bash
#SBATCH --job-name=clever-p6-matrix
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p6_matrix_%A_%a.log
#SBATCH --error=logs/p6_matrix_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 6: full experiment matrix.
#   3 datasets (lmsys, qqp, moss) × 2 embedding models (MiniLM-384,
#   gte-base-768) × 3 cache sizes (10/20/30%) = 18 array tasks.
#   Each task: 6 policies × 3 seeds (42/123/456) at 100k queries, temporal.
#   Total result cells: 3 × 2 × 6 × 3 × 3 seeds = 324 streams.
#
# PREREQUISITE: slurm/phase6_embed_lmsys.sbatch must have completed (it
# creates the nested LMSYS artifacts). Submit with a dependency:
#   EMBED=$(sbatch --parsable slurm/phase6_embed_lmsys.sbatch)
#   sbatch --dependency=afterok:$EMBED slurm/phase6_full_matrix.sbatch
#
# Array index → (dataset, model, cache):
#   idx = ds_idx*6 + model_idx*3 + cache_idx

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/eviction

DATASETS=(lmsys qqp moss)
MODELS=("all-MiniLM-L6-v2" "thenlper/gte-base")
MODEL_TAGS=(minilm gte)
CACHES=(0.10 0.20 0.30)
CACHE_TAGS=(c0p10 c0p20 c0p30)

TID="${SLURM_ARRAY_TASK_ID}"
DS="${DATASETS[$(( TID / 6 ))]}"
MODEL="${MODELS[$(( (TID / 3) % 2 ))]}"
MODEL_TAG="${MODEL_TAGS[$(( (TID / 3) % 2 ))]}"
CACHE="${CACHES[$(( TID % 3 ))]}"
CACHE_TAG="${CACHE_TAGS[$(( TID % 3 ))]}"
SIZE="100k"
OUT_DIR="results/eviction/phase6_matrix_${DS}_${MODEL_TAG}_${SIZE}_${CACHE_TAG}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== [$(date)] Phase 6 full matrix ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array ID: ${TID}"
echo "Node:     $(hostname)"
echo "Dataset:  $DS"
echo "Model:    $MODEL"
echo "Cache:    $CACHE"
echo "Output:   $OUT_DIR"
echo ""

python scripts/08_run_eviction.py \
    --dataset "$DS" \
    --embedding-model "$MODEL" \
    --size "$SIZE" \
    --config configs/eviction.yaml \
    --output "$OUT_DIR" \
    --policies lru lfu semantic arc gdsf siso \
    --cache-sizes "$CACHE" \
    --workloads temporal \
    --multi-seed

echo ""
echo "=== [$(date)] Phase 6 matrix task complete: $DS/$MODEL_TAG/$CACHE_TAG ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
