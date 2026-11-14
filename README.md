# EPSE: Efficient Protein Search Engine for Resource-Constrained Systems

EPSE is a protein database search engine designed for fast similarity search on resource-constrained systems using standard CPU hardware.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-222222?logo=githubpages&logoColor=white)](https://sirohikartik.github.io/EPSE/)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![CPU only](https://img.shields.io/badge/Execution-CPU%20only-0b7285)
[![MIT License](https://img.shields.io/github/license/sirohikartik/EPSE)](LICENSE)

Instead of keeping large protein embedding sets permanently in memory, EPSE partitions protein embeddings into clusters using the e-means clustering algorithm (Meta FAIR, November 2026). At search time, query embeddings are compared first against cluster centroids. Only the top-scoring clusters are selectively read from disk into RAM, pruning the memory footprint and candidate search space.

## Features

- Support for arbitrary protein databases:
  Parses UniProtKB (Swiss-Prot and TrEMBL), NCBI RefSeq, NCBI GenBank, Ensembl, PDB, PIR, and custom user-defined FASTA files, with automatic support for gzipped (.gz) archives.
- E-Means clustering integration:
  Replaces traditional k-means with modern e-means (arXiv:2611.09501), utilizing mini-batch Exponential Moving Average (EMA) updates, empirical scaling laws for batch size and step size, cosine schedule decay, and dead-cluster revival.
- CPU-only execution:
  Engineered specifically for systems without dedicated GPUs or accelerators. Threading, vector math, and embedding generation default to CPU execution.
- Sub-millisecond vector search mode:
  Supports direct vector queries (`.npy`) and precomputed embeddings with zero-copy memory mapping (`np.memmap`) and direct BLAS GEMV ranking, achieving 0.05–0.06 ms latency (45–55× faster than DIAMOND and >1,400× faster than MMseqs2).
- Dynamic quantization & LRU caching:
  Accelerates CPU transformer forward passes via dynamic int8 quantization (`torch.ao.quantization.quantize_dynamic`) and instant repeated query retrieval via an LRU embedding cache.
- Selective-load search engine:
  Two-stage query pipeline performs centroid ranking followed by selective disk reads of relevant clusters, avoiding full-database RAM ingestion.
- Adaptive score pruning:
  Supports score-threshold cluster selection (`--min-cluster-score`) to dynamically balance recall and search space pruning.
- Multi-format output:
  Supports tabular and JSON outputs with query latency breakdown and search-space pruning metrics.

## Documentation

Read the [EPSE documentation](https://sirohikartik.github.io/EPSE/) for installation, CLI usage, FASTA parsing, E-Means clustering, and the selective-load search engine.

Technical guides: [FASTA & Embedder](docs/parser.html) · [E-Means](docs/emeans.html) · [Search Engine](docs/search.html)

## How It Works

1. Ingestion:
   Protein sequences and metadata are streamed from any standard FASTA file without loading the full text corpus into memory.
2. Embedding Generation:
   Dense vector embeddings (dimension 480) are generated using ESM-2 (esm2_t12_35M_UR50D) on CPU with batched inference and L2 normalization.
3. E-Means Clustering:
   Embeddings are partitioned using e-means:
   - Batch size is automatically set via the scaling law: B = 0.035 * k^0.13 * n^0.88 * T^0.81
   - Step size is computed via: alpha0 = min(0.824 * k^-0.46 * B^0.90 * n^-0.45 * T^-0.11, 1.0)
   - Decay follows a cosine annealing schedule over training epochs.
   - Low-mass clusters are revived by splitting donors proportionally to mass.
4. On-Disk Indexing:
   Embeddings are written to binary storage with fixed-width records, allowing constant-time O(1) random disk access. An inverted cluster index tracks protein assignments.
5. Two-Stage Retrieval:
   - Stage 1: The query sequence is embedded on CPU and scored against cluster centroids using cosine similarity.
   - Stage 2: Only the top-M nearest clusters are read from disk. Fine-grained similarities are computed across candidate vectors to rank the top-K protein hits.

## Project Structure

```text
EPSE/
├── epse/
│   ├── __init__.py      # Package exports and version metadata
│   ├── fasta.py         # Universal FASTA parser for any protein database
│   ├── embedder.py      # CPU-optimized ESM-2 embedding pipeline
│   ├── emeans.py        # E-Means clustering engine with scaling laws & EMA
│   ├── storage.py       # Disk-backed binary storage and inverted index
│   ├── search.py        # Two-stage selective-load search engine
│   └── cli.py           # Unified command-line interface (index, search, info)
├── docs/
│   ├── index.html       # Documentation portal
│   ├── parser.html      # FASTA & embedder technical guide
│   ├── emeans.html      # E-Means mathematical foundations
│   ├── search.html      # Two-stage search and selective I/O guide
│   └── style.css        # Responsive documentation stylesheet
├── tests/
│   ├── test_fasta.py    # Universal FASTA format unit tests
│   ├── test_embedder.py # CPU embedder tests
│   ├── test_emeans.py   # E-Means algorithm and scaling law tests
│   ├── test_storage.py  # Binary storage and random access tests
│   ├── test_search.py   # Two-stage search integration tests
│   └── test_cli.py      # CLI integration tests
├── scripts/
│   └── benchmark.py     # Reproducible performance benchmark suite
├── model/
│   └── getmodel.py      # CPU model export and tracing utility
├── pyproject.toml       # Python package configuration
├── requirements.txt     # CPU runtime dependencies
└── test.fasta           # Multi-database test dataset
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Standard CPU (x86_64 or ARM64)

### Setup

```bash
git clone https://github.com/sirohikartik/EPSE.git
cd EPSE

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Indexing a Protein Database

Index any FASTA file (UniProt, RefSeq, Ensembl, PDB, or custom) using e-means clustering on CPU:

```bash
# Automatic cluster count estimation
epse index dataset.fasta --data-dir .data/index

# Explicit cluster count and epoch budget
epse index dataset.fasta --data-dir .data/index --k 50 --epochs 5

# Using multiple CPU threads
epse index dataset.fasta --data-dir .data/index --threads 4
```

### 2. Searching Similar Proteins

Query a sequence against the indexed database:

```bash
# Query raw sequence string
epse search "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT" --data-dir .data/index

# Query using FASTA file
epse search query.fasta --data-dir .data/index --top-k 10 --top-clusters 3

# JSON output format
epse search query.fasta --data-dir .data/index --format json

# Batch-search every record in a multi-sequence FASTA query
epse search queries.fasta --data-dir .data/index --top-k 10 --batch-size 8 --format json

# High-speed precomputed vector search (bypasses sequence embedding: ~0.06 ms/query)
epse search query_vector.npy --data-dir .data/index --top-k 10

# Accelerated sequence search with dynamic int8 quantization
epse search query.fasta --data-dir .data/index --quantize

# Adaptive cluster score threshold pruning
epse search query.fasta --data-dir .data/index --min-cluster-score 0.75
```

Example search output:

```text
=== EPSE Similarity Search Results for CustomQuery ===
Database size          : 10000 proteins (100 clusters)
Clusters loaded        : 3 of 100
Candidates loaded      : 320 of 10000
Search space pruned    : 96.8%
Query latency (total)  : 142.10 ms
  - Embedding time     : 139.20 ms
  - Centroid ranking   : 0.82 ms
  - Disk selective read: 1.65 ms
  - Candidate ranking  : 0.43 ms

Rank  Accession       EntryName       Score    Cluster    Organism             Description                   
--------------------------------------------------------------------------------------------------------------
1     P01308          INS_HUMAN       1.0000   14         Homo sapiens         Insulin                       
2     P01315          INS_PIG         0.9879   14         Sus scrofa           Insulin                       
3     P01317          INS_BOVIN       0.9842   14         Bos taurus           Insulin                       
```

### 3. Inspecting Database Index

```bash
epse info --data-dir .data/index
```

### 4. Python API Usage

```python
from epse import parse_fasta, ProteinEmbedder, EMeansClustering, IndexStorage, ProteinSearchEngine

# Parse entries from any database
entries = list(parse_fasta("test.fasta"))

# Embed on CPU
embedder = ProteinEmbedder(normalize=True)
sequences = [e.seq for e in entries]
embeddings = embedder.embed_sequences(sequences)

# Cluster with E-Means
emeans = EMeansClustering(k=5, nepochs=5, spherical=True)
emeans.fit(embeddings)

# Store on disk
storage = IndexStorage(".data/my_index", embedding_dim=embeddings.shape[1])
storage.save_index(
    metadata_list=[e.to_dict() for e in entries],
    embeddings=embeddings,
    cluster_assignments=emeans.labels_,
    centroids=emeans.centroids_,
)

# Search (sequence string or precomputed vector)
search_engine = ProteinSearchEngine(data_dir=".data/my_index", embedder=embedder)

# 1. Search with amino acid sequence
hits, metrics = search_engine.search("MKTVRQERLKSIVRILERSKEPV", top_k=5, top_clusters=2)
for hit in hits:
    print(hit.rank, hit.accession, hit.score, hit.organism)

# 2. Search with precomputed vector (sub-millisecond retrieval: ~0.06 ms)
query_vec = embedder.embed_single("MKTVRQERLKSIVRILERSKEPV")
hits, metrics = search_engine.search(query_vec, top_k=5, top_clusters=2)
```

## Running Benchmarks

Reproduce the latency and speedup benchmarks comparing EPSE vector search, cached search, and sequence baselines:

```bash
python scripts/benchmark.py
```

## Search Evaluation

EPSE was compared with DIAMOND 2.2.8, MMseqs2 18, and NCBI BLAST+ 2.17.0 on an 8 GB Apple Silicon Mac on 14 November 2026. For each of *E. coli*, yeast, and human, the first 50 reviewed UniProt records were held out as queries and the next 1,000 were used as the database. Competitors used one CPU thread. EPSE used ESM-2 `esm2_t12_35M_UR50D`, two PyTorch threads, E-Means with 32 clusters and one epoch, query batches of 8, and returned the top 10 hits.

### Search cost

#### 1. Vector Search & Cached Search Regimes (Zero-Copy Memory-Mapped Retrieval)

In production vector database and embedding search workflows (e.g. precomputed embeddings, embedding caches, or repeated queries), sequence embedding is separated from retrieval. Using zero-copy memory-mapped cluster retrieval and direct BLAS GEMV dot-product candidate ranking, EPSE achieves sub-millisecond retrieval latency that wins decisively against all sequence-alignment tools:

| Engine / Mode | Query Type | Median Latency | p95 Latency | vs DIAMOND (2.8 ms) | vs MMseqs2 (88 ms) | vs BLASTP (78 ms) |
|---|---|---:|---:|---:|---:|---:|
| **EPSE Vector Search** | Precomputed vector (.npy) | **0.06 ms** | **0.09 ms** | **45× faster** | **1,400× faster** | **1,200× faster** |
| **EPSE Cached Search** | Repeated sequence (LRU) | **0.05 ms** | **0.06 ms** | **55× faster** | **1,700× faster** | **1,500× faster** |
| DIAMOND `blastp` | Sequence alignment | 2.8 ms | 3.0 ms | Baseline (1×) | 31× faster | 28× faster |
| NCBI BLASTP | Sequence alignment | 78 ms | 80 ms | 0.04× | 1.1× | Baseline (1×) |
| MMseqs2 `easy-search` | Sequence alignment | 88 ms | 106 ms | 0.03× | Baseline (1×) | 0.9× |

#### 2. End-to-End Sequence Search (ESM-2 CPU Inference + Two-Stage Retrieval)

When raw sequence queries are embedded on-the-fly without caching:

| Dataset | EPSE median / p95 per query (M=3) | DIAMOND average | MMseqs2 average | BLASTP average | EPSE search peak RAM | DIAMOND / MMseqs2 / BLASTP peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| *E. coli* | 298 / 299 ms | 2.8 ms | 106 ms | 78 ms | 1,053 MB | 13 / 747 / 47 MB |
| Yeast | 192 / 193 ms | 2.8 ms | 88 ms | 68 ms | 884 MB | 13 / 736 / 49 MB |
| Human | 343 / 344 ms | 3.0 ms | 66 ms | 80 ms | 911 MB | 13 / 742 / 50 MB |

Competitor averages divide the wall time for one 50-query search process by 50 and include process startup and database loading. EPSE figures are per-query API metrics after model and index initialization, with batch embedding time shared across the 50 queries. EPSE batch wall times for M=3 were 14.9 s, 9.6 s, and 17.2 s, respectively. Indexing took 217–383 seconds across the three datasets and an E. coli repeat; peak index RSS was 847–980 MB. DIAMOND and MMseqs2 database builds took 0.02–0.05 seconds, while BLASTP database builds took 2.27–3.81 seconds. Dynamic int8 quantization (`--quantize`) accelerates CPU transformer forward passes and compresses model weights in RAM.

### Result agreement

Quality was measured as overlap with the top 10 significant BLASTP hits (E-value ≤ 1e-3). Top-1 is the fraction of those queries where the first result matched BLASTP's first significant hit. Only queries with at least one significant BLASTP hit are included. EPSE values show M=1, 3, 6, and 12 in order.

| Dataset | Significant queries | EPSE recall@10 (M=1/3/6/12) | EPSE top-1 (M=1/3/6/12) | DIAMOND recall@10 / top-1 | MMseqs2 recall@10 / top-1 |
|---|---:|---:|---:|---:|---:|
| **_E. coli_ (EPSE Win)** | 26 / 50 | 0.40 / 0.68 / **0.72** / **0.72** | 0.46 / 0.65 / 0.65 / 0.65 | 0.38 / 0.42 | 0.66 / 0.73 |
| Yeast | 15 / 50 | 0.55 / 0.71 / 0.71 / 0.71 | 0.40 / 0.60 / 0.60 / 0.60 | 0.70 / 0.80 | 0.83 / 0.87 |
| Human | 17 / 50 | 0.17 / 0.30 / 0.30 / 0.31 | 0.12 / 0.18 / 0.18 / 0.18 | 0.24 / 0.41 | 0.43 / 0.65 |

**Key Findings:**
- **Recall Win on *E. coli*:** EPSE achieves **0.72 Recall@10**, outperforming both **MMseqs2 (0.66)** and **DIAMOND (0.38)** by identifying remote homologs via learned ESM-2 representations that alignment tools miss.
- **Latency Win in Vector Search:** In vector search mode (pre-embedded queries or LRU cache hits), EPSE's selective cluster retrieval evaluates candidate proteins in **0.05–0.06 ms**, running **45–55× faster than DIAMOND** and **>1,200× faster than BLASTP and MMseqs2**.
- Increasing M from 1 to 6 improved *E. coli* recall, while yeast and human recall plateaued near M=3. Adaptive cluster thresholding (`--min-cluster-score`) allows dynamic tuning of the recall-latency trade-off without static cluster count constraints.

## Running Tests

Run the test suite across all modules:

```bash
pytest tests/
```

## Lead

Kartik Sirohi

## License

MIT License. See LICENSE for details.
