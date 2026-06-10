#!/bin/bash
#SBATCH --job-name=clever-p1-evict-smoke
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1_eviction_smoke_%A_%a.log
#SBATCH --error=logs/p1_eviction_smoke_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 1c: Smoke-test the new 100k MOSS/QQP MiniLM embeddings through the
# eviction harness. This is intentionally narrow: LRU/LFU only, 10% cache,
# temporal workload, one seed. It validates that Phase 1 data and embeddings
# are consumable before moving to density profiling.

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/eviction

DATASETS=(qqp moss)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL="all-MiniLM-L6-v2"
SIZE="100k"
OUT_DIR="results/eviction/phase1_smoke_${DS}_minilm_${SIZE}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== [$(date)] Phase 1 eviction smoke ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID:-local}"
echo "Node:     $(hostname)"
echo "Dataset:  $DS"
echo "Model:    $MODEL"
echo "Size:     $SIZE"
echo "Output:   $OUT_DIR"
echo ""

python scripts/08_run_eviction.py \
    --dataset "$DS" \
    --embedding-model "$MODEL" \
    --size "$SIZE" \
    --config configs/eviction.yaml \
    --output "$OUT_DIR" \
    --policies lru lfu \
    --cache-sizes 0.10 \
    --workloads temporal

echo ""
echo "=== [$(date)] Phase 1 eviction smoke complete: $DS ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
