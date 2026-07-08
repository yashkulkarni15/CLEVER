# CLEVER — Cluster-Level Eviction for Vector Embedding Retrieval

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Benchmarking and characterization framework for semantic caching in LLM applications.

**Course:** CSE 584 — Advanced Database Systems, University of Michigan
**Team:** Yash Kulkarni, Shubham Harkare, Arvind Suresh

## Overview

CLEVER evaluates the three layers of a semantic LLM cache — ANN indexing, cost-based query routing, and cache eviction — under realistic workloads. Its primary contribution is a rigorous **negative result with a diagnostic**: semantic-aware eviction yields no hit-rate benefit over LFU on sparse real-world workloads while adding 3–8× latency overhead, and a formal workload **density metric** explains when frequency-based policies gain and when semantic redundancy is too weak to exploit. An **LLM-judge audit** then shows the raw hit rates themselves are largely illusory: at the standard similarity threshold, most cache hits are not semantically equivalent queries, and quality-adjusting the hit rate collapses it by an order of magnitude on the sparse workloads.

Experiments span **3 datasets** (LMSYS-Chat-1M, Quora Question Pairs, MOSS) × **2 embedding models** (MiniLM-L6-v2 384-d, gte-base 768-d, with per-encoder threshold calibration) × **6 eviction policies** (LRU, LFU, Semantic, ARC, GDSF, SISO) × **3 cache sizes**, all multi-seed (42, 123, 456), plus a 13.7K-pair LLM-judged hit-quality evaluation.

The paper lives in this repo: source at `main.tex`, compiled draft at `paper/main.pdf`.

## Datasets

| Dataset | Role | Source | Scale |
|---------|------|--------|-------|
| LMSYS-Chat-1M | Sparse/diverse conversational | `lmsys/lmsys-chat-1m` | 579,753 unique queries (full) + 100K subset |
| Quora Question Pairs | Clustered near-duplicates | `quora` | 537K processed, 100K subset |
| MOSS | Saturated control (heavy repetition) | `fnlp/moss-002-sft-data` | 100K subset (English, first turn) |

Per-dataset artifacts follow the convention `data/<dataset>/<size>_queries.parquet` and `results/embeddings/<dataset>/<model_tag>/<size>_embeddings.npy` (see `src/data/paths.py`).

## Key Results

### ANN Index Benchmarking (499K vectors)

| Index | Recall@1 | P50 Latency | QPS |
|-------|----------|-------------|-----|
| Flat (exact) | 1.000 | 17.39 ms | 58 |
| IVF (nlist=1024) | 1.000 | 1.10 ms | 907 |
| **HNSW (M=32, ef=128)** | **0.996** | **0.56 ms** | **11,056** |
| LSH | 0.745 | — | — |

HNSW is the Pareto winner: 31× faster than flat search with only 0.4% recall loss. Locked in for all downstream experiments.

### Cost-Based Query Routing (579K stream, 5 seeds)

The router thresholds on FAISS **L2² distance** between unit-norm embeddings (`cosine = 1 − L2²/2`); a query is a hit when `L2² ≤ θ`.

- **Random cache fill:** θ = 0.772 ± 0.015 (cosine ≥ 0.61) → **60.4% hit rate and 60.4% monetary savings**, accepted-hit cosine quality ≈ 0.806
- **Frequency-based fill:** θ = 0.760 (cosine ≥ 0.62) → 54.5% hit rate — popular topics over-consume slots, reducing coverage diversity

### Eviction at Full Scale — the Negative Result (LMSYS 579K, 3 seeds)

| Cache | LRU | LFU | Semantic | Overhead (Sem) |
|-------|-----|-----|----------|----------------|
| 10% | 0.7802 | **0.7835** | 0.7808 | 8.5 ms/query (6.1× LRU) |
| 20% | 0.8453 | 0.8453 | **0.8453** | 12.5 ms/query (7.4× LRU) |
| 30% | 0.8756 | **0.8756** | 0.8756 | 15.9 ms/query (8.4× LRU) |

On sparse workloads the embedding space has near-uniform density: nearly every entry looks equally isolated, the redundancy signal r(e) ≈ 0, and the semantic score collapses to LRU+LFU ordering — all cost, no benefit.

### Cross-Dataset Baseline Comparison (100K, 10% cache, 3 seeds)

| Policy | LMSYS | QQP | MOSS | Stream time vs LRU |
|--------|-------|-----|------|--------------------|
| LRU | 0.5554 | 0.5750 | 0.9761 | 1.0× |
| **LFU** | **0.5701** | **0.5999** | 0.9761 | 1.3–1.4× |
| Semantic | 0.5594 | 0.5874 | 0.9761 | 3.8–4.3× |
| ARC | 0.5701 | 0.5999 | 0.9761 | 2.4–2.5× |
| GDSF | 0.5680 | 0.5951 | 0.9760 | 2.0× |
| SISO | 0.5066 | 0.5144 | 0.9761 | 3.1–3.5× |

**LFU is undefeated across the board.** ARC converges to LFU's decisions on sparse workloads. SISO — the closest prior work on semantic-locality eviction — is the *worst* policy on both sparse datasets, churning on locality signals that do not exist. MOSS saturates (~0.976) for every policy and serves as a control. Lightweight adaptive policies (hard-switch / blended-score over LRU/LFU/Semantic) were also evaluated and do not beat LFU; the hard switch simply learns to select LFU.

### Workload Density Characterization

`src/profiler/density.py` measures active-cache density `r(e) = fraction of cached entries within L2² θ of e` during eviction runs. LFU's gains over LRU coincide with higher cache density on QQP (+2.49 pp) and LMSYS (+1.47 pp) and vanish on saturated MOSS. Note that the metric measures *cache-content* density: a dense workload self-deduplicates (near-duplicates are served as hits and never inserted), so cache density can invert raw workload density.

### Cross-Encoder Replication (gte-base)

Re-running the matrix under gte-base with the MiniLM-calibrated hit threshold (L2² 0.90) is **degenerate**: gte-base is anisotropic (random-pair cosine ≈ 0.69–0.76), so every query "hits", eviction never fires, and all policies report a perfect 100% hit rate. After per-encoder recalibration (`configs/eviction_gte.yaml`: hit threshold 0.30 L2² anchored at the gte NN-distance median, with SISO/ARC thresholds moved to the same quantiles), the policy **ordering replicates exactly** — ARC = LFU everywhere, SISO worst on the sparse datasets, all gaps shrinking with capacity — but absolute hit rates do not transfer (LMSYS LFU 76.0% under gte vs 57.0% under MiniLM; QQP runs the other direction). Semantic-cache thresholds must be recalibrated per encoder.

### LLM-Judged Hit Quality (quality-adjusted hit rate)

Every logged cache hit pairs the new query with the cached query it matched. A stratified sample (1,000 hits per dataset×policy cell, 13,669 unique pairs) is judged for query equivalence by a local `llama3.1:8b` (Ollama, temperature 0), cross-validated against Groq's hosted `llama-3.1-8b-instant` on a 1,931-pair overlap (98.55% agreement, Cohen's κ = 0.826).

| Dataset | Raw hit rate | Judged-equivalent | Quality-adjusted |
|---------|--------------|-------------------|------------------|
| LMSYS | 50.7–57.0% | 3.1–3.9% | **1.6–2.2%** |
| QQP | 51.4–60.0% | 2.1–3.1% | **1.1–1.8%** |
| MOSS | 97.6% | 24.7–27.0% | **24.1–26.4%** |

**Most raw hits are not answers**, and after quality adjustment the differences between eviction policies fall within binomial noise (Wilson 95% CIs overlap in every dataset). Equivalence decays with hit distance, and near-zero embedding distance does not imply equivalence: MiniLM saturates on long templated prompts, so even LMSYS's nearest-decile hits (median L2² 0.10) are only 8.9% equivalent. The operative lever in a semantic cache is the hit threshold and match signal — not the eviction policy.

## Quick Start

### 1. Setup Environment

**Option A — pip (recommended):**
```bash
git clone https://github.com/yashkulkarni15/CLEVER.git
cd CLEVER
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option B — Conda:**
```bash
conda env create -f environment.yaml
conda activate clever
```

**Verify installation:**
```bash
bash scripts/00_setup_environment.sh
pytest tests/ -m "not integration" -v
```

### 2. Download & Preprocess Data

```bash
# LMSYS-Chat-1M (requires HuggingFace login)
huggingface-cli login
python scripts/01_download_dataset.py --output data/ --max-rows 10000   # dev mode
python scripts/01_download_dataset.py --output data/                   # full

# QQP / MOSS (multi-dataset pipeline)
python scripts/01_download_dataset.py --dataset qqp  --raw-path data/raw/qqp/quora_duplicate_questions.tsv
python scripts/01_download_dataset.py --dataset moss --raw-path data/raw/moss/ --all-turns
```

### 3. Generate Embeddings

```bash
# Local (CPU, small subsets for development)
python scripts/02_generate_embeddings.py \
    --input data/processed_queries.parquet \
    --output results/embeddings/ \
    --device cpu --batch-size 64 \
    --sizes 10k,50k

# Per-dataset / per-model (nested output, never overwrites)
python scripts/02_generate_embeddings.py \
    --input data/qqp/processed_queries.parquet \
    --dataset qqp --model thenlper/gte-base \
    --device cuda --batch-size 512 --sizes 100k

# Great Lakes (GPU)
sbatch slurm/phase1_embed.sbatch
```

### 4. Run an Eviction Experiment

```bash
python scripts/08_run_eviction.py \
    --dataset qqp --embedding-model all-MiniLM-L6-v2 --size 100k \
    --config configs/eviction.yaml \
    --output results/eviction/my_run/ \
    --policies lru lfu semantic arc gdsf siso \
    --cache-sizes 0.10 --workloads temporal --multi-seed
```

### 5. Run the LLM-Judge Pipeline

```bash
# Log every cache hit during an eviction run (adds hits_seed42.jsonl per run dir)
python scripts/08_run_eviction.py ... --log-hits

# Stratified sample: 1,000 hits per (dataset, policy) cell
python scripts/16_sample_hits_for_judge.py \
    --hits-glob "results/eviction/phase7_loghits_*_minilm_100k_c0p10/hits_*.jsonl" \
    --out results/judge/phase7_sample_n1000.jsonl --n 1000

# Judge with any OpenAI-compatible endpoint (cp .env.example .env first;
# local Ollama: JUDGE_BASE_URL=http://localhost:11434/v1, JUDGE_MODEL=llama3.1:8b)
python scripts/17_run_llm_judge.py \
    --sample results/judge/phase7_sample_n1000.jsonl \
    --out-dir results/judge/my_judge_run \
    --raw-results-glob "results/eviction/phase7_loghits_*_minilm_100k_c0p10/eviction_results.json" \
    --raw-cache-key 0.10

# Figures + LaTeX table from the verdicts
python scripts/18_visualize_judge.py --judge-dir results/judge/my_judge_run
```

The judge runner is crash-safe and resumable: verdicts append to `verdict_cache.jsonl` as they arrive, cached pairs are never re-submitted, and `--max-calls` caps spend on metered APIs.

## Project Structure

```
CLEVER/
├── src/                     # Core library
│   ├── data/                # Loaders (LMSYS/MOSS/QQP), preprocessing, sampling, path resolver
│   ├── embeddings/          # Sentence-transformers encoder (MiniLM, gte-base)
│   ├── indexes/             # FAISS index wrappers (Flat, HNSW, IVF, LSH)
│   ├── cache/               # Semantic cache + eviction policies:
│   │   └── eviction/        #   LRU, LFU, Semantic, ARC, GDSF, SISO, adaptive (ablation), Oracle
│   ├── router/              # Cost-based adaptive query routing
│   ├── profiler/            # Workload density profiler
│   ├── judge/               # LLM judge: config (.env), OpenAI-compatible client, rubric
│   ├── benchmark/           # Metrics, workload generation, index profiling
│   ├── evaluation/          # Routing evaluator, analysis
│   └── utils/               # Manifest generation, environment checks, results parsing
├── scripts/                 # CLI entry points (one per experiment)
├── configs/                 # YAML experiment configurations (per-encoder: eviction.yaml, eviction_gte.yaml)
├── slurm/                   # Great Lakes HPC job scripts
├── tests/                   # pytest suite (232 passing)
├── main.tex                 # Paper source (compiled draft: paper/main.pdf)
├── paper/                   # Compiled draft, figures, and tables used by the paper
├── results/                 # Outputs (embeddings, benchmarks, density, judge, figures)
├── data/                    # Processed datasets (parquet)
├── requirements.txt         # Pinned pip dependencies
└── environment.yaml         # Conda environment specification
```

## Experiment Scripts

| Script | Description |
|--------|-------------|
| `00_setup_environment.sh` | Verify environment |
| `01_download_dataset.py` | Download + preprocess (LMSYS / MOSS / QQP via `--dataset`) |
| `01b_generate_synthetic_data.py` | Generate synthetic embeddings at scale |
| `02_generate_embeddings.py` | Encode queries (`--dataset`/`--model` nest outputs) |
| `03_run_index_benchmark.py` | Index comparison (HNSW, IVF, LSH, Flat) |
| `04`–`05_visualize_*.py` | Dataset EDA + benchmark plots |
| `06_run_routing_eval.py` | Cost-based query routing evaluation |
| `07_visualize_routing.py` | Routing evaluation plots |
| `08_run_eviction.py` | Eviction harness — all 6 policies, multi-seed, multi-cache-size |
| `09_visualize_eviction.py` | Eviction result visualizations |
| `10`–`11_*_semantic.py` | Semantic policy parameter sweep + ablation |
| `12_run_density_profile.py` | Workload density profiling during eviction runs |
| `13_run_adaptive_subset.py` | Adaptive hard-switch / blended-score comparison |
| `14_visualize_phase_figures.py` | Density characterization + adaptive comparison figures |
| `15_visualize_phase6_matrix.py` | Cross-encoder matrix figures (cache-size ablation, policy heatmap) |
| `16_sample_hits_for_judge.py` | Stratified hit sampling for the LLM judge |
| `17_run_llm_judge.py` | Resumable LLM-judge runner (any OpenAI-compatible endpoint) |
| `18_visualize_judge.py` | Quality-adjusted hit-rate figures + LaTeX table |

### Slurm Jobs (Great Lakes HPC)

| Job | Script | Resources |
|-----|--------|-----------|
| Embeddings (legacy LMSYS) | `slurm/embed.sbatch` | 1× GPU, 32 GB |
| Multi-dataset data prep | `slurm/phase1_data.sbatch` | CPU, 32 GB |
| Multi-dataset embeddings | `slurm/phase1_embed.sbatch` | 1× GPU, 48 GB |
| Index benchmarks | `slurm/benchmark.sbatch` | CPU, 32 GB |
| Routing | `slurm/routing.sbatch` | CPU, 32 GB |
| Eviction (full LMSYS) | `slurm/eviction.sbatch` | largemem, 16 CPUs, 64 GB |
| Density profiling | `slurm/phase2_density.sbatch` | CPU array (6), 48 GB |
| Adaptive subset | `slurm/phase3_adaptive_subset.sbatch` | CPU array (3), 48 GB |
| Baseline comparison | `slurm/phase4_baselines.sbatch` | CPU array (3), 48 GB |
| LMSYS gte-base embeddings | `slurm/phase6_embed_lmsys.sbatch` | 1× GPU, 48 GB |
| Full experiment matrix | `slurm/phase6_full_matrix.sbatch` | CPU array (18), 64 GB |
| gte-base recalibrated re-run | `slurm/phase6_gte_recal.sbatch` | CPU array (9), 64 GB |
| Hit logging for the judge | `slurm/phase7_log_hits.sbatch` | CPU array (3), 48 GB |

## Eviction Policies

The semantic eviction policy scores each cached entry as:

```
score(e) = (r(e) + μ) / utility(e)
```

where:
- `r(e)` = redundancy — fraction of e's semantic neighbors currently in cache
- `utility(e)` = α · recency + β · frequency + ε (recency floor prevents immortality)
- `μ` = additive smoothing constant; as μ → ∞ the ordering asymptotically approaches plain inverse-utility (LRU+LFU behaviour)

The entry with the highest score is evicted. Semantic neighbors are maintained via an incremental symmetric redundancy graph (updated on every insert/evict) with periodic full rebuilds every `recompute_interval` evictions.

**Baselines** (all subclass `src/cache/eviction/base.py:EvictionPolicy`):
- **ARC** — classic adaptive replacement cache (T1/T2/B1/B2 with adaptive p); ghost hits detected semantically (evicted-embedding match) since cache ids are slot-unique
- **GDSF** — greedy-dual size-frequency with inflation clock; on unit-norm fixed-dimension embeddings it reduces to LFU-with-aging
- **SISO** — faithful implementation of Kim et al., *Rethinking Caching for LLM Serving Systems* (arXiv:2508.18736): cluster-size/access-count eviction, new-region protection, maintenance decay, M/D/1-driven dynamic threshold
- **Adaptive (ablation)** — hard-switch and blended-score policies driven by online frequency-skew and density signals; both collapse to LFU in practice

**Configuration** (`configs/eviction.yaml`):

```yaml
eviction:
  semantic:
    similarity_threshold: 0.90   # L2² threshold; must be ≤ hit_threshold
    alpha: 1.0                   # recency weight
    beta: 1.0                    # frequency weight
    mu: 0.1                      # additive smoothing
    dynamic_impute: true         # incremental graph updates on insert
    recompute_interval: 2000     # full graph rebuild every N evictions
```

`similarity_threshold` must satisfy `≤ hit_threshold` so that two entries are semantic neighbours if and only if one would be a cache hit for the other.

## Running Tests

```bash
pytest tests/ -m "not integration" -v
```

232 tests cover the six eviction policies (correctness, edge cases, paper-faithful behaviors), adaptive policies, the density profiler, dataset loaders, path resolution, multi-seed reproducibility, hit logging, the judge sampler/client/runner, and the figure data layers. Integration tests (network-bound model downloads) are deselected for fast local runs.

On macOS, pin threads to avoid a FAISS/OpenMP segfault:
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -m "not integration" -q
```

## Hardware Requirements

- **Local (MacBook):** Development, tests, ≤50K vectors. CPU-only FAISS.
- **Great Lakes HPC:** Full experiments (100K–580K). GPU for embeddings; CPU/largemem partitions for eviction evaluation (~19 hours for the full-scale 3-seed LMSYS matrix).

## Reproducibility

All stochastic components use deterministic seeding. Multi-seed evaluation (seeds 42, 123, 456) reports mean ± std; variance across seeds is near-zero (std ≤ 0.0002 on hit rate), confirming stable evaluation. Every result file includes a manifest with hardware specs, Git state, and configuration hash via `src/utils/manifest.py`. Subset sampling is deterministic per seed, so dataset subsets are identical across embedding models.

## License

MIT
