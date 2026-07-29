#!/bin/bash
#SBATCH --job-name=clever-p10-fifo
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p10_fifo_%A_%a.log
#SBATCH --error=logs/p10_fifo_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 10: FIFO baseline across the full Phase 6 matrix.
#
# Sun, Cao & Lam (arXiv:2607.00394) report that LRU and LFU consistently
# underperform a naive FIFO baseline on semantic workloads, attributing it to
# absent temporal locality and frequency concentration. Their setting is agent
# retrieval buffers (LoCoMo, DialSim); ours is query-level response caching on
# conversational serving traces. This job adds the missing control so the two
# settings can be compared directly rather than argued about.
#
#   3 datasets (lmsys, qqp, moss) × 2 encoders (minilm, gte) × 3 cache sizes
#   = 18 array tasks, matching the Phase 6 grid exactly.
#   Each task: FIFO only × 3 seeds (42/123/456) at 100k queries, temporal.
#   Array index → idx = ds_idx*6 + model_idx*3 + cache_idx  (same as phase6).
#
# Config per encoder mirrors the canonical Phase 6 cells: MiniLM uses
# configs/eviction.yaml, gte uses configs/eviction_gte.yaml (the recalibrated
# per-encoder thresholds). Comparing FIFO against the DEGENERATE
# phase6_matrix_*_gte_* cells would be meaningless; compare against
# phase6_recal_*_gte_* instead.
#
# FIFO is O(1) per eviction and holds no embedding state, so these are the
# cheapest cells in the study. Results land in NEW dirs and merge into the
# existing matrix at analysis time by policy key (results JSON is keyed
# per-policy under both `per_seed` and `aggregated`).
#
# PREREQUISITE: 100k embeddings for both encoders already exist on GLC from
# Phase 6. No re-embedding needed. Submit directly:
#   sbatch slurm/phase10_fifo.sbatch

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/eviction

DATASETS=(lmsys qqp moss)
MODELS=("all-MiniLM-L6-v2" "thenlper/gte-base")
MODEL_TAGS=(minilm gte)
CONFIGS=(configs/eviction.yaml configs/eviction_gte.yaml)
CACHES=(0.10 0.20 0.30)
CACHE_TAGS=(c0p10 c0p20 c0p30)
SIZE="100k"

TID="${SLURM_ARRAY_TASK_ID}"
DS="${DATASETS[$(( TID / 6 ))]}"
MODEL_IDX=$(( (TID / 3) % 2 ))
MODEL="${MODELS[$MODEL_IDX]}"
MODEL_TAG="${MODEL_TAGS[$MODEL_IDX]}"
CONFIG="${CONFIGS[$MODEL_IDX]}"
CACHE="${CACHES[$(( TID % 3 ))]}"
CACHE_TAG="${CACHE_TAGS[$(( TID % 3 ))]}"
OUT_DIR="results/eviction/phase10_fifo_${DS}_${MODEL_TAG}_${SIZE}_${CACHE_TAG}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== [$(date)] Phase 10 FIFO baseline ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array ID: ${TID}"
echo "Node:     $(hostname)"
echo "Dataset:  $DS"
echo "Model:    $MODEL"
echo "Config:   $CONFIG"
echo "Cache:    $CACHE"
echo "Output:   $OUT_DIR"
echo ""

python scripts/08_run_eviction.py \
    --dataset "$DS" \
    --embedding-model "$MODEL" \
    --size "$SIZE" \
    --config "$CONFIG" \
    --output "$OUT_DIR" \
    --policies fifo \
    --cache-sizes "$CACHE" \
    --workloads temporal \
    --multi-seed

echo ""
echo "=== [$(date)] Phase 10 FIFO task complete: $DS/$MODEL_TAG/$CACHE_TAG ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
