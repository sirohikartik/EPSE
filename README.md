# EPSE: Efficient Protein Search Engine for Resource-Constrained Systems

EPSE is an experimental protein database search engine focused on reducing memory usage while maintaining fast similarity search over large protein datasets.

Instead of loading all protein embeddings into memory, EPSE groups proteins into clusters based on embedding similarity. At query time, only the most relevant clusters are loaded into RAM, reducing both memory consumption and search space.

For more information visit : https://epse.vercel.app


## Documentation
Project documentation: https://epse-docs.vercel.app

## Current Status
Implemented components:
- FASTA file parsing
- Protein embedding generation using ESM-2 (t12, 35M)
- Protein clustering using cosine similarity
- Cluster centroid generation (`centroids.csv`)
- Disk-based storage of embeddings
- CMake build system

The search pipeline is currently under active development.

## How It Works
1. **Protein embeddings** are generated using the ESM-2 (t12, 35M parameter) protein language model.
2. **Clustering**: Rather than loading every protein embedding into memory, EPSE organizes proteins into clusters. Each cluster is represented by a centroid (the average embedding of the proteins assigned to that cluster).
3. **Query Process**:
    - The query embedding is generated.
    - The embedding is compared against cluster centroids using cosine similarity.
    - The most relevant clusters are selected.
    - Only proteins from those clusters are loaded into RAM.
    - Search is performed within this reduced search space.

This approach aims to reduce memory requirements while maintaining efficient retrieval.

## Project Structure
```text
EPSE/
├── docs/
│   └── index.html
├── model/
│   ├── esm2_t12_35M_tokenizer/
│   └── getmodel.py
├── parser/
│   ├── embedder.cpp
│   ├── embedder.h
│   └── parse.c
├── CMakeLists.txt
├── README.md
├── .clang-format
└── .gitignore
```

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

Currently only the parser has been implemented, so to run do - ./epse from the project root.
The system currently expects the uniprot_sprot.fasta file in the project root.
Future versions will support configurable dataset locations.


## Contributions

Contributors are welcome.

```text
Getting Started

    Fork the repository.

    Create a feature branch: git checkout -b my-feature

    Make your changes.

    Commit your work: git commit -m "Describe your change"

    Push your branch: git push origin my-feature

    Open a Pull Request.
```


Project Lead : Kartik Sirohi

## License

License information will be added in a future release.
