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
- Selective-load search engine:
  Two-stage query pipeline performs centroid ranking followed by selective disk reads of relevant clusters, avoiding full-database RAM ingestion.
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

# Search
search_engine = ProteinSearchEngine(data_dir=".data/my_index", embedder=embedder)
hits, metrics = search_engine.search("MKTVRQERLKSIVRILERSKEPV", top_k=5, top_clusters=2)
for hit in hits:
    print(hit.rank, hit.accession, hit.score, hit.organism)
```

## Search Evaluation

EPSE was compared with DIAMOND 2.2.8, MMseqs2 18, and NCBI BLAST+ 2.17.0 on an 8 GB Apple Silicon Mac on 14 November 2026. The benchmark used the first 500 reviewed UniProt records returned for each of three organisms (*E. coli*, yeast, and human): the first 20 records were held out as queries, and the remaining 480 formed the search database. Search tools used one CPU thread; EPSE used ESM-2 `esm2_t12_35M_UR50D`, two PyTorch threads, E-Means with 20 clusters and one epoch, and returned the top 10 hits after loading three clusters.

### Search cost

| Tool | Search time per query | Peak search RAM |
|---|---:|---:|
| EPSE | 54–224 ms median | 495–551 MB |
| DIAMOND `blastp` | 4–6 ms average | 12–13 MB |
| MMseqs2 `easy-search` | 243–353 ms average | 711–728 MB |
| NCBI BLASTP | 166–182 ms average | 46–48 MB |

Competitor averages divide the wall time for a batch of 20 queries by 20; that time includes process startup and database loading. EPSE latency is the median per-query API time after model and index initialization, and includes query embedding. EPSE's 95th-percentile query latency ranged from 419 to 824 ms. Indexing took 205–224 seconds and peaked at 654–664 MB on the yeast and human subsets; those index timings were not retained for *E. coli*.

### Result agreement

Quality was measured as overlap with the top 10 BLASTP hits having E-value ≤ 1e-3. Top-1 is the fraction of those queries where the first result matched BLASTP's first significant hit. Only queries with at least one significant BLASTP hit are included.

| Dataset | Significant queries | EPSE recall@10 / top-1 | DIAMOND recall@10 / top-1 | MMseqs2 recall@10 / top-1 |
|---|---:|---:|---:|---:|
| *E. coli* | 6 / 20 | 0.67 / 0.67 | 0.42 / 0.50 | 0.83 / 0.83 |
| Yeast | 5 / 20 | 0.94 / 0.80 | 0.82 / 1.00 | 0.94 / 1.00 |
| Human | 6 / 20 | 0.45 / 0.17 | 0.34 / 0.83 | 0.47 / 0.83 |

These are small, exploratory subsets, and BLASTP is the reference rather than independent biological ground truth. EPSE ranks embedding similarity while the other tools perform sequence alignment, so their scores are not directly equivalent. This run shows that EPSE fits within 8 GB at this database size, but does not establish how its selective loading performs on large databases. Search time and memory can change substantially with database size, query composition, and tool settings.

## Running Tests

Run the test suite across all modules:

```bash
pytest tests/
```

## Lead

Kartik Sirohi

## License

MIT License. See LICENSE for details.
