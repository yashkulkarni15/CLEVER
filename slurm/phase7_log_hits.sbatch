#!/bin/bash
#SBATCH --job-name=p7_loghits
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/p7_loghits_%A_%a.log
#SBATCH --error=slurm/logs/p7_loghits_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

DATASETS=(lmsys qqp moss)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUT="results/eviction/phase7_loghits_${DS}_minilm_100k_c0p10"

COMMON=(--config configs/eviction.yaml --policies lru lfu semantic arc gdsf siso
        --cache-sizes 0.10 --output "$OUT" --log-hits --dataset "$DS")

if [ "$DS" = "lmsys" ]; then
  # LMSYS predates the Option-A nested layout → explicit legacy overrides.
  python scripts/08_run_eviction.py "${COMMON[@]}" \
    --embeddings results/embeddings/100k_embeddings.npy \
    --queries data/100k_queries.parquet
else
  python scripts/08_run_eviction.py "${COMMON[@]}" \
    --embedding-model all-MiniLM-L6-v2 --size 100k
fi
echo "DONE ${DS}"
