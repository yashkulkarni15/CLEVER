#!/bin/bash
#SBATCH --job-name=p7_loghits
#SBATCH --account=stats_dept2
#SBATCH --partition=standard
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/p7_loghits_%A_%a.log
#SBATCH --error=logs/p7_loghits_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Phase 7 prep: re-run the 10% MiniLM cells with --log-hits so every cache hit
# dumps its (orig_query, new_query, distance) pair for the LLM-judge sampler.
# 3 datasets x 6 policies x 10% cache, single seed (42), temporal workload.
#
# Submit from the slurm/ directory (like every other phase job): the relative
# --output path lands the logs in slurm/logs/, and PROJECT_ROOT below is
# absolute so the run itself is submission-directory independent.

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p results/eviction

DATASETS=(lmsys qqp moss)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUT="results/eviction/phase7_loghits_${DS}_minilm_100k_c0p10"

python scripts/08_run_eviction.py \
  --config configs/eviction.yaml \
  --policies lru lfu semantic arc gdsf siso \
  --cache-sizes 0.10 \
  --output "$OUT" \
  --log-hits \
  --dataset "$DS" \
  --embedding-model all-MiniLM-L6-v2 \
  --size 100k

echo "DONE ${DS}"
