#!/bin/bash
#SBATCH --job-name=clever-p1-embed
# NOTE: stats_dept1/stats_dept2 have NO GPU allocation (AssocGrpGRES if used for
# GPU jobs). Use engin1 for GPU (gpu=2, cpu=36, mem=180G). (cse584w26_class also
# has a GPU but is NOT available for this project's use.)
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/p1_embed_%j.log
#SBATCH --error=logs/p1_embed_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# ──────────────────────────────────────────────────────────────
# Phase 1b: Generate the 2×2 embedding matrix (MOSS, QQP) × (MiniLM,
# gte-base). GPU job. Requires Phase 1a parquets to exist first.
# Outputs nest under results/embeddings/<dataset>/<model_tag>/ so
# nothing overwrites the existing LMSYS embeddings.
#
# PRE-FETCH MODELS ONCE on a login node (compute nodes may lack
# internet) so the HF cache is warm:
#   source venv/bin/activate
#   python -c "from sentence_transformers import SentenceTransformer as S; \
#     S('all-MiniLM-L6-v2'); S('thenlper/gte-base')"
# ──────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"
source venv/bin/activate
mkdir -p logs results/embeddings

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

SIZES="10k,50k,100k,full"

for DS in qqp moss; do
  for MODEL in all-MiniLM-L6-v2 thenlper/gte-base; do
    echo "=== [$(date)] embeddings: dataset=$DS model=$MODEL ==="
    python scripts/02_generate_embeddings.py \
        --input "data/$DS/processed_queries.parquet" \
        --dataset "$DS" --model "$MODEL" \
        --device cuda --batch-size 512 --sizes "$SIZES" \
        --skip-existing
  done
done

echo "=== [$(date)] Phase 1b done. Tree: ==="
find results/embeddings -name '*_embeddings.npy' | sort
