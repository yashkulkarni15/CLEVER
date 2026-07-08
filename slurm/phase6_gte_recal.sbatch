#!/bin/bash
#SBATCH --job-name=clever-p6-gte-recal
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p6_gte_recal_%A_%a.log
#SBATCH --error=logs/p6_gte_recal_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 6 RE-RUN: gte-base cells only, with per-encoder recalibrated thresholds.
#
# The original phase6_full_matrix gte cells were DEGENERATE (every policy 100.00%
# hit / 0 evictions) because they used the MiniLM-tuned hit_threshold (0.90 L2² =
# cos 0.55), which on anisotropic gte-base admits ~all pairs. This job re-runs the
# 9 gte cells with configs/eviction_gte.yaml (hit_threshold 0.30 / cos 0.85, plus
# recalibrated semantic/SISO/ARC thresholds — see that file's header). The MiniLM
# half is unaffected and is NOT re-run.
#
#   3 datasets (lmsys, qqp, moss) × 3 cache sizes (10/20/30%) = 9 array tasks.
#   Each task: 6 policies × 3 seeds (42/123/456) at 100k queries, temporal.
#   Array index → (dataset, cache):  idx = ds_idx*3 + cache_idx
#
# Results go to NEW dirs (phase6_recal_*) so the original degenerate cells are
# preserved as before/after evidence for the calibration methodology point.
#
# PREREQUISITE: gte-base 100k embeddings already exist on GLC (the original
# phase6_full_matrix run produced them). No re-embedding needed. Submit directly:
#   sbatch slurm/phase6_gte_recal.sbatch

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/eviction

DATASETS=(lmsys qqp moss)
CACHES=(0.10 0.20 0.30)
CACHE_TAGS=(c0p10 c0p20 c0p30)
MODEL="thenlper/gte-base"
MODEL_TAG="gte"
SIZE="100k"

TID="${SLURM_ARRAY_TASK_ID}"
DS="${DATASETS[$(( TID / 3 ))]}"
CACHE="${CACHES[$(( TID % 3 ))]}"
CACHE_TAG="${CACHE_TAGS[$(( TID % 3 ))]}"
OUT_DIR="results/eviction/phase6_recal_${DS}_${MODEL_TAG}_${SIZE}_${CACHE_TAG}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== [$(date)] Phase 6 gte recalibration re-run ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array ID: ${TID}"
echo "Node:     $(hostname)"
echo "Dataset:  $DS"
echo "Model:    $MODEL  (config: configs/eviction_gte.yaml)"
echo "Cache:    $CACHE"
echo "Output:   $OUT_DIR"
echo ""

python scripts/08_run_eviction.py \
    --dataset "$DS" \
    --embedding-model "$MODEL" \
    --size "$SIZE" \
    --config configs/eviction_gte.yaml \
    --output "$OUT_DIR" \
    --policies lru lfu semantic arc gdsf siso \
    --cache-sizes "$CACHE" \
    --workloads temporal \
    --multi-seed

echo ""
echo "=== [$(date)] Phase 6 gte-recal task complete: $DS/$MODEL_TAG/$CACHE_TAG ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
