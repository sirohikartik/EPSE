# EPSE: Efficient Protein Search Engine for Resource-Constrained Systems

EPSE is a protein database search engine designed for fast similarity search on resource-constrained systems using standard CPU hardware.

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
- Complete backward compatibility:
  Maintains backward compatibility with earlier EPSE dataset formats and the parser entrypoint.

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
├── parser/
│   ├── __init__.py
│   ├── parse.py         # Backward-compatible script wrapper
│   └── legacy/          # Preserved C and C++ source implementations
├── tests/
│   ├── test_fasta.py    # Universal FASTA format unit tests
│   ├── test_embedder.py # CPU embedder tests
│   ├── test_emeans.py   # E-Means algorithm and scaling law tests
│   ├── test_storage.py  # Binary storage and random access tests
│   ├── test_search.py   # Two-stage search integration tests
│   ├── test_cli.py      # CLI integration tests
│   └── test_parse.py    # Compatibility pipeline tests
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

### 4. Backward-Compatible Parser Script

The original script invocation remains supported:

```bash
python parser/parse.py 0.90 dataset.fasta
```

### 5. Python API Usage

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

## Running Tests

Run the test suite across all modules:

```bash
pytest tests/
```

## Lead

Kartik Sirohi

## License

MIT License. See LICENSE for details.
