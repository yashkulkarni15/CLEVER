#!/bin/bash
#SBATCH --job-name=clever-p3-adaptive
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/p3_adaptive_%A_%a.log
#SBATCH --error=logs/p3_adaptive_%A_%a.err

set -euo pipefail

cd /home/yashkulk/scratch/yashkulk/CSE584/CLEVER
source venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p logs results/adaptive

DATASETS=(lmsys qqp moss)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL="all-MiniLM-L6-v2"
SIZE="100k"
MAX_QUERIES="${MAX_QUERIES:-30000}"
OUT="results/adaptive/phase3_subset_${DATASET}_minilm_${MAX_QUERIES}"

echo "=== [$(date)] Phase 3A adaptive subset ==="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Node:        $(hostname)"
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL}"
echo "Size:        ${SIZE}"
echo "Max queries: ${MAX_QUERIES}"
echo "Output:      ${OUT}"
echo

if [[ "${DATASET}" == "lmsys" ]]; then
  python scripts/13_run_adaptive_subset.py \
    --embeddings results/embeddings/100k_embeddings.npy \
    --queries data/100k_queries.parquet \
    --dataset lmsys \
    --embedding-model "${MODEL}" \
    --size "${SIZE}" \
    --config configs/eviction.yaml \
    --output "${OUT}" \
    --policies lru lfu semantic adaptive_hard adaptive_blend \
    --cache-size 0.10 \
    --workload temporal \
    --max-queries "${MAX_QUERIES}"
else
  python scripts/13_run_adaptive_subset.py \
    --dataset "${DATASET}" \
    --embedding-model "${MODEL}" \
    --size "${SIZE}" \
    --config configs/eviction.yaml \
    --output "${OUT}" \
    --policies lru lfu semantic adaptive_hard adaptive_blend \
    --cache-size 0.10 \
    --workload temporal \
    --max-queries "${MAX_QUERIES}"
fi

echo
echo "=== [$(date)] Phase 3A adaptive subset complete: ${DATASET} ==="
ls -lh "${OUT}"/adaptive_subset_*.csv
