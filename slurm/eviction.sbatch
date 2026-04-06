#!/bin/bash
#SBATCH --job-name=clever-eviction
#SBATCH --account=stats_dept2
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/eviction_%j.log
#SBATCH --error=logs/eviction_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ──────────────────────────────────────────────────────────────
# Phase 4: Semantic-Aware Eviction Policy Evaluation (Multi-Seed)
# largemem partition — more cores + memory for FAISS batch search.
# Runs the full eviction matrix: 4 policies x 3 cache sizes x 3 seeds
#
# Supports checkpoint/resume: if the job is killed mid-run,
# resubmit and it will skip already-completed runs automatically.
#
# IMPORTANT: Before first submission, create the logs directory:
#   mkdir -p ~/scratch/yashkulk/CSE584/CLEVER/logs
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== CLEVER Phase 4: Eviction Policy Evaluation (Multi-Seed, largemem) ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Memory:   ${SLURM_MEM_PER_NODE:-128000} MB"
echo "Started:  $(date)"
echo ""

# ── Setup ────────────────────────────────────────────────────
PROJECT_ROOT=~/scratch/yashkulk/CSE584/CLEVER
cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

# Create output directories
mkdir -p results/eviction results/figures logs

# ── Verify faiss-cpu is installed (no GPU needed) ────────────
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')" || {
    echo ">>> Installing faiss-cpu..."
    pip install faiss-cpu
}

# ── Pin numpy threads to match allocated CPUs ────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ">>> OMP/MKL threads pinned to $SLURM_CPUS_PER_TASK"
echo ""

# ── Run eviction experiments (multi-seed, all policies) ──────
# Config drives the full matrix:
#   policies: [lru, lfu, semantic, oracle]
#   cache_sizes_pct: [0.10, 0.20, 0.30]
#   seeds: [42, 123, 456]
#   workloads: [temporal]
#   oracle.refresh_interval: 5000
#   oracle.horizon: 50000 (bounded lookahead)
#   log_interval: 500
#
# Checkpoint/resume: if a previous run was interrupted, this will
# automatically skip completed runs and resume from where it left off.
echo ">>> Running multi-seed eviction evaluation (largemem, checkpoint-enabled)..."
python scripts/08_run_eviction.py \
    --embeddings results/embeddings/full_embeddings.npy \
    --queries data/full_queries.parquet \
    --config configs/eviction.yaml \
    --output results/eviction/ \
    --multi-seed

echo ""
echo ">>> Eviction evaluation complete."

# ── Generate visualizations ────────────────────────────────────
echo ">>> Generating eviction visualizations (8 figures)..."
python scripts/09_visualize_eviction.py \
    --results results/eviction/eviction_results_multi_seed.json \
    --output results/figures/

echo ""
echo "=== Phase 4 Complete ==="
echo "Finished: $(date)"
echo "Results:  results/eviction/"
echo "Figures:  results/figures/"
