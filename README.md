# CLEVER — Cluster-Level Eviction for Vector Embedding Retrieval

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Benchmarking and optimization framework for semantic caching in LLM applications.

**Course:** CSE 584 — Advanced Database Systems, University of Michigan  
**Team:** Yash Kulkarni, Shubham Harkare, Arvind Suresh

## Overview

CLEVER evaluates ANN index structures (HNSW, IVF, LSH, Flat) for semantic caching under realistic LLM workloads. It implements cost-based query routing, proposes a semantic-aware eviction policy with additive smoothing and an incremental redundancy graph, and analyzes scalability from 10K to 1M cached entries. All experiments run on the full LMSYS-Chat-1M dataset (579,753 unique queries, 384-dim MiniLM-L6-v2 embeddings) with multi-seed evaluation (seeds 42, 123, 456).

## Key Results

### Phase 2 — ANN Index Benchmarking (499K vectors)

| Index | Recall@1 | P50 Latency | QPS |
|-------|----------|-------------|-----|
| Flat (exact) | 1.000 | 17.39 ms | 58 |
| IVF (nlist=1024) | 1.000 | 1.10 ms | 907 |
| **HNSW (M=32, ef=128)** | **0.996** | **0.56 ms** | **11,056** |
| LSH | 0.745 | — | — |

HNSW is the Pareto winner: 31× faster than flat search with only 0.4% recall loss. IVF achieves full recall but 12× fewer QPS. LSH is not viable at this scale.

### Phase 3 — Cost-Based Query Routing (579K stream)

- **Random cache fill:** 59.1% hit rate at threshold θ = 0.76 with 80.3% semantic quality (cosine ≥ 0.8) — 60.3% latency savings vs. always calling the LLM
- **Frequency-based fill:** 54.6% hit rate — lower because popular topics over-consume slots, reducing coverage diversity
- **Sweet spot:** θ ∈ [0.7, 0.9] balances hit rate and semantic quality; below 0.7 quality degrades, above 0.9 hits drop sharply

### Phase 4 — Eviction Policy Evaluation (3 seeds × 3 policies × 3 cache sizes)

| Cache | LRU | LFU | Semantic | Overhead (Sem) |
|-------|-----|-----|----------|----------------|
| 10% | 0.7802 | **0.7835** | 0.7808 | 8.5 ms/query (6.1× LRU) |
| 20% | 0.8453 | 0.8453 | **0.8453** | 12.5 ms/query (7.4× LRU) |
| 30% | 0.8756 | **0.8756** | 0.8756 | 15.9 ms/query (8.4× LRU) |

The semantic eviction policy (score = (r + μ) / utility, with incremental symmetric redundancy graph and μ-smoothing) is algorithmically correct and prevents isolated-entry immortality, but yields no hit rate improvement on this workload. LMSYS is a high-diversity conversational dataset — the average L2² distance between any two queries is large, so semantic neighbor graphs are sparse and the redundancy signal r ≈ 0 for most entries, causing the policy to degrade to LRU+LFU ordering. On clustered workloads (repeated coding questions, math problems), the policy would be expected to outperform. The μ-smoothing parameter ensures safe fallback to inverse-utility ordering as μ → ∞.

## Quick Start

### 1. Setup Environment

**Option A — pip (recommended):**
```bash
git clone <repo-url>
cd CLEVER
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option B — Conda:**
```bash
git clone <repo-url>
cd CLEVER
conda env create -f environment.yaml
conda activate clever
```

**Verify installation:**
```bash
bash scripts/00_setup_environment.sh
pytest tests/ -v
```

### 2. Download & Preprocess Data

```bash
# Login to HuggingFace (required for LMSYS-Chat-1M access)
huggingface-cli login

# Download and preprocess (dev mode — first 10K rows)
python scripts/01_download_dataset.py --output data/ --max-rows 10000

# Full dataset (run on Great Lakes or with patience)
python scripts/01_download_dataset.py --output data/
```

### 3. Generate Embeddings

```bash
# Local (CPU, small subsets for development)
python scripts/02_generate_embeddings.py \
    --input data/processed_queries.parquet \
    --output results/embeddings/ \
    --device cpu --batch-size 64 \
    --sizes 10k,50k

# Great Lakes (GPU, all subsets)
sbatch slurm/embed.sbatch
```

## Project Structure

```
CLEVER/
├── src/                     # Core library
│   ├── data/                # Data loading, preprocessing, sampling
│   ├── embeddings/          # Sentence-transformers encoder
│   ├── indexes/             # FAISS index wrappers (Flat, HNSW, IVF, LSH)
│   ├── cache/               # Semantic cache + eviction policies (LRU, LFU, Semantic, Oracle)
│   ├── router/              # Cost-based adaptive query routing
│   ├── benchmark/           # Metrics, workload generation, profiling
│   ├── evaluation/          # Routing evaluator, analysis
│   └── utils/               # Manifest generation, environment checks
├── scripts/                 # CLI entry points (one per phase)
├── configs/                 # YAML experiment configurations
├── slurm/                   # Great Lakes HPC job scripts
├── tests/                   # pytest test suite (42 eviction tests)
├── results/                 # Outputs (embeddings, benchmarks, figures 00–35)
├── data/                    # Processed datasets (parquet)
├── requirements.txt         # Pinned pip dependencies
└── environment.yaml         # Conda environment specification
```

## Experiment Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `00_setup_environment.sh` | Verify environment |
| 1a | `01_download_dataset.py` | Download + preprocess LMSYS-Chat-1M |
| 1b | `01b_generate_synthetic_data.py` | Generate synthetic embeddings at scale |
| 1c | `02_generate_embeddings.py` | Encode queries → 384-dim embeddings |
| 2 | `03_run_index_benchmark.py` | Index comparison (HNSW, IVF, LSH, Flat) |
| — | `04_visualize_data.py` | Dataset EDA visualizations (Figs 04–10) |
| — | `05_visualize_benchmarks.py` | Benchmark result plots (Figs 11–18) |
| 3 | `06_run_routing_eval.py` | Cost-based query routing evaluation |
| — | `07_visualize_routing.py` | Routing evaluation plots (Figs 21–27) |
| 4 | `08_run_eviction.py` | Eviction policy evaluation (LRU, LFU, Semantic) — multi-seed |
| — | `09_visualize_eviction.py` | Eviction result visualizations (Figs 28–35) |
| — | `10_tune_semantic.py` | Local parameter sweep: similarity_threshold × μ grid |
| — | `11_ablation_semantic.py` | Ablation: threshold progression + μ variants |

### Slurm Jobs (Great Lakes HPC)

| Job | Script | Resources |
|-----|--------|-----------|
| Embeddings | `slurm/embed.sbatch` | 1× GPU, 32 GB |
| Benchmarks | `slurm/benchmark.sbatch` | CPU, 32 GB |
| Routing | `slurm/routing.sbatch` | CPU, 32 GB |
| Eviction | `slurm/eviction.sbatch` | largemem, 16 CPUs, 64 GB |

## Eviction Policy Design

The semantic eviction policy scores each cached entry as:

```
score(e) = (r(e) + μ) / utility(e)
```

where:
- `r(e)` = redundancy — fraction of e's semantic neighbors currently in cache
- `utility(e)` = α · recency + β · frequency + ε (recency floor prevents immortality)
- `μ` = additive smoothing constant; as μ → ∞ the ordering asymptotically approaches plain inverse-utility (LRU+LFU behaviour)

The entry with the highest score is evicted. Semantic neighbors are maintained via an incremental symmetric redundancy graph (updated on every insert/evict) with periodic full rebuilds every `recompute_interval` evictions.

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
pytest tests/ -v
```

42 eviction tests cover LRU, LFU, semantic scoring, μ-smoothing, graph consistency, and multi-seed reproducibility.

## Hardware Requirements

- **Local (MacBook):** Development, tests, ≤50K vectors. CPU-only FAISS.
- **Great Lakes HPC:** Full experiments (100K–580K). GPU for embeddings; largemem partition (16 CPUs, 64 GB) for eviction evaluation (~19 hours for full 3-seed matrix).

## Reproducibility

All stochastic components use deterministic seeding. Multi-seed evaluation (seeds 42, 123, 456) reports mean ± std. Variance across seeds is near-zero (std ≤ 0.00005 on hit rate), confirming stable evaluation. Every result file includes a manifest with hardware specs, Git state, and configuration hash via `src/utils/manifest.py`.

## License

MIT
