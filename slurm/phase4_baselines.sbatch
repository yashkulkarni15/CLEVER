#!/bin/bash
#SBATCH --job-name=clever-p4-baselines
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p4_baselines_%A_%a.log
#SBATCH --error=logs/p4_baselines_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 4: Baseline comparison. Runs the two classic baselines (ARC, GDSF)
# and the closest prior work (SISO, Kim et al. arXiv:2508.18736) against
# LRU/LFU/Semantic on all three datasets: 100k MiniLM, 10% cache, temporal
# workload, multi-seed (42/123/456 from configs/eviction.yaml).

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/eviction

DATASETS=(lmsys qqp moss)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL="all-MiniLM-L6-v2"
SIZE="100k"
OUT_DIR="results/eviction/phase4_baselines_${DS}_minilm_${SIZE}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== [$(date)] Phase 4 baseline comparison ==="
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
    --policies lru lfu semantic arc gdsf siso \
    --cache-sizes 0.10 \
    --workloads temporal \
    --multi-seed

echo ""
echo "=== [$(date)] Phase 4 baseline comparison complete: $DS ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
