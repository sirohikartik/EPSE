"""Unified Command Line Interface for EPSE.

Provides commands to index protein databases using e-means clustering on CPU,
perform fast selective-load similarity search, and inspect index statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import numpy as np

from .fasta import parse_fasta, UniversalFastaEntry
from .embedder import ProteinEmbedder, DEFAULT_MODEL_NAME
from .emeans import EMeansClustering
from .storage import IndexStorage
from .search import ProteinSearchEngine


def cmd_index(args: argparse.Namespace) -> int:
    """Builds EPSE index from a FASTA file using e-means clustering."""
    fasta_path = args.fasta
    data_dir = args.data_dir or f".data/k_{args.k or 'auto'}"

    print("=== EPSE Indexing Pipeline (CPU-Optimized E-Means) ===")
    print(f"Dataset path  : {fasta_path}")
    print(f"Output directory : {data_dir}")
    print(f"E-Means epochs: {args.epochs}")
    print(f"Clustering k  : {args.k if args.k else 'Automatic (sqrt scaling)'}")
    print(f"Model name    : {args.model_name}")
    print(f"CPU threads   : {args.threads if args.threads else 'Default'}")
    print()

    if not os.path.exists(fasta_path):
        print(f"ERROR: FASTA file not found: {fasta_path}", file=sys.stderr)
        return 1

    # Count once so the embedding matrix can be allocated as a disk-backed
    # memmap without retaining all sequence strings in memory.
    t0 = time.perf_counter()
    print("Counting protein records...")
    n_entries = sum(1 for _ in parse_fasta(fasta_path))
    t_parse = time.perf_counter() - t0
    print(f"Parsed {n_entries} protein records across database in {t_parse:.2f}s.")

    if n_entries == 0:
        print("ERROR: No valid protein sequences found in FASTA file.", file=sys.stderr)
        return 1

    # 2. Embed records in batches directly into a temporary memmap.
    t0 = time.perf_counter()
    print(f"Generating embeddings on CPU for {n_entries} sequences...")
    embedder = ProteinEmbedder(
        model_name=args.model_name,
        num_threads=args.threads,
        normalize=True,
        use_mock=args.use_mock,
        quantize=getattr(args, "quantize", False),
    )
    if args.batch_size <= 0:
        print("ERROR: batch size must be positive.", file=sys.stderr)
        return 1

    os.makedirs(data_dir, exist_ok=True)
    fd, embedding_tmp = tempfile.mkstemp(prefix=".epse-embeddings-", dir=data_dir)
    os.close(fd)
    embeddings = None
    try:
        embeddings = np.memmap(
            embedding_tmp,
            mode="w+",
            dtype=np.float32,
            shape=(n_entries, embedder.embedding_dim),
        )
        metadata_list = []
        pending_sequences = []
        write_start = 0

        def flush_batch() -> None:
            nonlocal write_start
            if not pending_sequences:
                return
            batch_embeddings = embedder.embed_sequences(
                pending_sequences, batch_size=args.batch_size
            )
            embeddings[write_start : write_start + len(pending_sequences)] = batch_embeddings
            write_start += len(pending_sequences)
            pending_sequences.clear()

        for entry in parse_fasta(fasta_path):
            metadata_list.append(
                {
                    "accession": entry.accession,
                    "entry_name": entry.entry_name,
                    "description": entry.description,
                    "organism": entry.organism,
                    "seq_len": len(entry.seq),
                }
            )
            pending_sequences.append(entry.seq)
            if len(pending_sequences) >= args.batch_size:
                flush_batch()
        flush_batch()

        if write_start != n_entries:
            raise RuntimeError(
                f"FASTA record count changed during indexing ({n_entries} then {write_start})."
            )
        embeddings.flush()
        t_embed = time.perf_counter() - t0
        print(f"Generated {n_entries} embeddings (dim={embeddings.shape[1]}) in {t_embed:.2f}s.")

        # 3. Cluster normalized embeddings without making a full in-memory copy.
        t0 = time.perf_counter()
        print("Clustering embeddings with E-Means (EMA mini-batch updates + scaling laws)...")
        emeans = EMeansClustering(
            k=args.k,
            nepochs=args.epochs,
            spherical=True,
            seed=args.seed,
        )
        emeans.fit(embeddings, assume_normalized=True)
        t_cluster = time.perf_counter() - t0
        print(f"E-Means clustered {n_entries} proteins into {emeans.k} centroids in {t_cluster:.2f}s.")

        # 4. Save final index files in bounded chunks.
        t0 = time.perf_counter()
        print(f"Saving database artifacts to {data_dir}...")
        storage = IndexStorage(data_dir=data_dir, embedding_dim=embeddings.shape[1])
        storage.save_index(
            metadata_list=metadata_list,
            embeddings=embeddings,
            cluster_assignments=emeans.labels_,
            centroids=emeans.centroids_,
            cluster_counts=emeans.cluster_sizes_,
        )
        t_save = time.perf_counter() - t0

        print(f"Index successfully written in {t_save:.2f}s.")
        print("Files created:")
        print(f"  - Metadata   : {storage.proteins_csv}")
        print(f"  - Embeddings : {storage.embeddings_bin}")
        print(f"  - Assignments: {storage.assignments_bin}")
        print(f"  - Centroids  : {storage.centroids_csv}")
        print(f"  - Inverted idx: {storage.cluster_index_path}")
        print(f"\nDone! Total pipeline time: {(t_parse + t_embed + t_cluster + t_save):.2f}s.\n")
        return 0
    finally:
        if embeddings is not None:
            embeddings.flush()
            del embeddings
        try:
            os.remove(embedding_tmp)
        except FileNotFoundError:
            pass


def cmd_search(args: argparse.Namespace) -> int:
    """Executes similarity search for a protein query."""
    data_dir = args.data_dir
    query_input = args.query

    if not os.path.exists(data_dir):
        print(f"ERROR: Index directory not found: {data_dir}", file=sys.stderr)
        return 1

    query_entries = None
    query_vectors = None
    # Check if query is a file path or a raw sequence
    if os.path.exists(query_input):
        if query_input.endswith(".npy"):
            loaded_arr = np.load(query_input)
            if loaded_arr.ndim == 1:
                query_seq = loaded_arr.astype(np.float32)
                query_label = os.path.splitext(os.path.basename(query_input))[0]
            elif loaded_arr.ndim == 2:
                query_vectors = [loaded_arr[i].astype(np.float32) for i in range(len(loaded_arr))]
                query_label = os.path.splitext(os.path.basename(query_input))[0]
            else:
                print(f"ERROR: .npy query must be 1D or 2D array, got shape {loaded_arr.shape}", file=sys.stderr)
                return 1
        else:
            query_entries = list(parse_fasta(query_input))
            if not query_entries:
                print(f"ERROR: Could not parse query FASTA file: {query_input}", file=sys.stderr)
                return 1
            query_seq = query_entries[0].seq
            query_label = query_entries[0].accession
    else:
        query_seq = query_input.strip().upper()
        query_label = "CustomQuery"

    embedder = None
    if query_vectors is None and (isinstance(query_seq, str) or query_entries is not None):
        embedder = ProteinEmbedder(
            model_name=args.model_name,
            num_threads=args.threads,
            normalize=True,
            use_mock=args.use_mock,
            quantize=getattr(args, "quantize", False),
        )

    search_engine = ProteinSearchEngine(data_dir=data_dir, embedder=embedder)
    if query_vectors is not None:
        batch_results = search_engine.search_many(
            query_vectors,
            top_k=args.top_k,
            top_clusters=args.top_clusters,
            batch_size=args.batch_size,
            min_cluster_score=args.min_cluster_score,
        )
        if args.format == "json":
            output = {
                "queries": [
                    {
                        "query": f"Vector_{idx + 1}",
                        "metrics": {
                            "total_proteins": metrics.total_proteins,
                            "candidates_loaded": metrics.candidates_loaded,
                            "clusters_loaded": metrics.clusters_loaded,
                            "search_space_pruned_pct": round(metrics.search_space_pruned_pct, 2),
                            "total_time_ms": round(metrics.total_time_sec * 1000, 2),
                        },
                        "hits": [hit.to_dict() for hit in hits],
                    }
                    for idx, (hits, metrics) in enumerate(batch_results)
                ]
            }
            print(json.dumps(output, indent=2))
            return 0

        for idx, (hits, metrics) in enumerate(batch_results, start=1):
            print(
                f"Vector_{idx}: {metrics.total_time_sec * 1000:.2f} ms, "
                f"{metrics.candidates_loaded}/{metrics.total_proteins} candidates"
            )
            for hit in hits:
                print(f"  {hit.rank:>3}  {hit.accession:<15} {hit.score:.4f}")
        return 0

    if query_entries is not None and len(query_entries) > 1:
        batch_results = search_engine.search_many(
            [entry.seq for entry in query_entries],
            top_k=args.top_k,
            top_clusters=args.top_clusters,
            batch_size=args.batch_size,
            min_cluster_score=args.min_cluster_score,
        )
        if args.format == "json":
            output = {
                "queries": [
                    {
                        "query": entry.accession,
                        "metrics": {
                            "total_proteins": metrics.total_proteins,
                            "candidates_loaded": metrics.candidates_loaded,
                            "clusters_loaded": metrics.clusters_loaded,
                            "search_space_pruned_pct": round(metrics.search_space_pruned_pct, 2),
                            "total_time_ms": round(metrics.total_time_sec * 1000, 2),
                        },
                        "hits": [hit.to_dict() for hit in hits],
                    }
                    for entry, (hits, metrics) in zip(query_entries, batch_results)
                ]
            }
            print(json.dumps(output, indent=2))
            return 0

        for entry, (hits, metrics) in zip(query_entries, batch_results):
            print(
                f"{entry.accession}: {metrics.total_time_sec * 1000:.2f} ms, "
                f"{metrics.candidates_loaded}/{metrics.total_proteins} candidates"
            )
            for hit in hits:
                print(f"  {hit.rank:>3}  {hit.accession:<15} {hit.score:.4f}")
        return 0

    results, metrics = search_engine.search(
        query=query_seq,
        top_k=args.top_k,
        top_clusters=args.top_clusters,
        min_cluster_score=args.min_cluster_score,
    )

    if args.format == "json":
        output = {
            "query": query_label,
            "metrics": {
                "total_proteins": metrics.total_proteins,
                "candidates_loaded": metrics.candidates_loaded,
                "clusters_loaded": metrics.clusters_loaded,
                "search_space_pruned_pct": round(metrics.search_space_pruned_pct, 2),
                "total_time_ms": round(metrics.total_time_sec * 1000, 2),
            },
            "hits": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
        return 0

    print(f"\n=== EPSE Similarity Search Results for {query_label} ===")
    print(f"Database size          : {metrics.total_proteins} proteins ({metrics.clusters_evaluated} clusters)")
    print(f"Clusters loaded        : {metrics.clusters_loaded} of {metrics.clusters_evaluated}")
    print(f"Candidates loaded      : {metrics.candidates_loaded} of {metrics.total_proteins}")
    print(f"Search space pruned    : {metrics.search_space_pruned_pct:.1f}%")
    print(f"Query latency (total)  : {metrics.total_time_sec * 1000:.2f} ms")
    print(f"  - Embedding time     : {metrics.embed_time_sec * 1000:.2f} ms")
    print(f"  - Centroid ranking   : {metrics.centroid_search_time_sec * 1000:.2f} ms")
    print(f"  - Disk selective read: {metrics.disk_load_time_sec * 1000:.2f} ms")
    print(f"  - Candidate ranking  : {metrics.ranking_time_sec * 1000:.2f} ms\n")

    if not results:
        print("No matching candidates found.")
        return 0

    header_fmt = "{:<5} {:<15} {:<15} {:<8} {:<10} {:<20} {:<30}"
    row_fmt = "{:<5} {:<15} {:<15} {:<8.4f} {:<10} {:<20} {:<30}"

    print(header_fmt.format("Rank", "Accession", "EntryName", "Score", "Cluster", "Organism", "Description"))
    print("-" * 110)
    for r in results:
        desc = (r.description[:27] + "...") if len(r.description) > 30 else r.description
        org = (r.organism[:17] + "...") if len(r.organism) > 20 else r.organism
        print(row_fmt.format(r.rank, r.accession, r.entry_name, r.score, r.cluster_id, org, desc))
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Displays information and statistics about an existing index."""
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory not found: {data_dir}", file=sys.stderr)
        return 1

    storage = IndexStorage(data_dir)
    try:
        centroids, counts = storage.load_centroids()
        cluster_index = storage.load_cluster_index()
    except Exception as e:
        print(f"ERROR reading index: {e}", file=sys.stderr)
        return 1

    total_prots = sum(len(pids) for pids in cluster_index.values())
    k, dim = centroids.shape

    print(f"\n=== EPSE Database Index Info ({data_dir}) ===")
    print(f"Total proteins indexed : {total_prots}")
    print(f"Total clusters (k)     : {k}")
    print(f"Embedding dimension    : {dim}")
    print(f"Min cluster size       : {min(counts)}")
    print(f"Max cluster size       : {max(counts)}")
    print(f"Mean cluster size      : {np.mean(counts):.1f}")
    print(f"Median cluster size    : {np.median(counts):.1f}")

    if os.path.exists(storage.embeddings_bin):
        size_bytes = os.path.getsize(storage.embeddings_bin)
        print(f"Embeddings file size   : {size_bytes / (1024 * 1024):.2f} MB")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="epse",
        description="EPSE: Efficient Protein Search Engine using E-Means on CPU",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index sub-command
    p_index = subparsers.add_parser("index", help="Index a FASTA protein database with e-means")
    p_index.add_argument("fasta", help="Path to input FASTA file (.fasta, .fa, .gz)")
    p_index.add_argument("--data-dir", "-d", default=".data", help="Output directory for index artifacts")
    p_index.add_argument("--k", "-k", type=int, default=None, help="Number of clusters (default: auto)")
    p_index.add_argument("--epochs", "-e", type=int, default=5, help="Number of e-means epochs (default: 5)")
    p_index.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for embedding generation")
    p_index.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="ESM-2 model name")
    p_index.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    p_index.add_argument("--seed", type=int, default=1234, help="Random seed for clustering")
    p_index.add_argument("--quantize", action="store_true", help="Apply dynamic int8 quantization on CPU")
    p_index.add_argument("--use-mock", action="store_true", help="Use deterministic mock embeddings for testing")
    p_index.set_defaults(func=cmd_index)

    # search sub-command
    p_search = subparsers.add_parser("search", help="Search a protein sequence against an index")
    p_search.add_argument("query", help="Query amino acid sequence, FASTA file, or .npy embedding vector file")
    p_search.add_argument("--data-dir", "-d", default=".data", help="Directory of indexed database")
    p_search.add_argument("--top-k", "-k", type=int, default=5, help="Number of hits to return (default: 5)")
    p_search.add_argument("--top-clusters", "-c", type=int, default=3, help="Number of clusters to load (default: 3)")
    p_search.add_argument("--min-cluster-score", type=float, default=None, help="Minimum centroid score threshold to load a cluster")
    p_search.add_argument("--format", choices=["table", "json"], default="table", help="Output display format")
    p_search.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="ESM-2 model name")
    p_search.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    p_search.add_argument("--batch-size", type=int, default=8, help="Sequences per embedding batch for multi-record FASTA queries")
    p_search.add_argument("--quantize", action="store_true", help="Apply dynamic int8 quantization on CPU")
    p_search.add_argument("--use-mock", action="store_true", help="Use deterministic mock embeddings for testing")
    p_search.set_defaults(func=cmd_search)

    # info sub-command
    p_info = subparsers.add_parser("info", help="Display index statistics")
    p_info.add_argument("--data-dir", "-d", default=".data", help="Directory of indexed database")
    p_info.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
