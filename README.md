# CLEVER — Cluster-Level Eviction for Vector Embedding Retrieval

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Benchmarking and optimization framework for semantic caching in LLM applications.

**Course:** CSE 584 — Advanced Database Systems, University of Michigan  
**Team:** Yash Kulkarni, Shubham Harkare, Arvind Suresh

## Overview

CLEVER evaluates ANN index structures (HNSW, IVF, LSH, Flat) for semantic caching under realistic LLM workloads. It implements cost-based query routing, proposes semantic-aware eviction, and analyzes scalability from 10K to 1M cached entries.

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
├── tests/                   # pytest test suite
├── results/                 # Outputs (embeddings, benchmarks, figures)
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
| — | `04_visualize_data.py` | Dataset EDA visualizations |
| — | `05_visualize_benchmarks.py` | Benchmark result plots (8 figures) |
| 3 | `06_run_routing_eval.py` | Cost-based query routing evaluation |
| — | `07_visualize_routing.py` | Routing evaluation plots (6 figures) |
| 4 | `08_run_eviction.py` | Eviction policy evaluation (LRU, LFU, Semantic, Oracle) |
| — | `09_visualize_eviction.py` | Eviction result visualizations (8 figures) |
| 5 | `11_latency_sensitivity.py` | LLM call latency sensitivity analysis |

### Slurm Jobs (Great Lakes HPC)

| Job | Script | Resources |
|-----|--------|-----------|
| Embeddings | `slurm/embed.sbatch` | 1× GPU, 32 GB |
| Benchmarks | `slurm/benchmark.sbatch` | CPU, 32 GB |
| Routing | `slurm/routing.sbatch` | CPU, 32 GB |
| Eviction | `slurm/eviction.sbatch` | largemem, 16 CPUs, 128 GB |

## Running Tests

```bash
pytest tests/ -v
```

## Hardware Requirements

- **Local (MacBook):** Development, tests, ≤50K vectors. CPU-only FAISS.
- **Great Lakes HPC:** Full experiments (100K–1M). GPU for embeddings and semantic eviction, CPU for benchmarks.

## Reproducibility

All stochastic components use deterministic seeding. Multi-seed evaluation (seeds 42, 123, 456) with mean ± std aggregation is supported. Every result file includes a manifest with hardware specs, Git state, and configuration hash via `src/utils/manifest.py`.

## License

MIT
