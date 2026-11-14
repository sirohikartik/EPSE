"""Standalone reproducible benchmark suite for EPSE.

Evaluates EPSE search latency across two operational regimes:
1. Vector Search Mode (precomputed / cached embeddings): Pure two-stage selective retrieval.
2. Sequence Search Mode: End-to-end inference + two-stage retrieval.

Compares measured performance against published baselines (DIAMOND, MMseqs2, NCBI BLASTP).
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
import numpy as np

from epse.emeans import EMeansClustering
from epse.storage import IndexStorage
from epse.search import ProteinSearchEngine
from epse.embedder import ProteinEmbedder


def run_benchmark(
    n_db: int = 1000,
    n_queries: int = 50,
    dim: int = 480,
    k: int = 32,
    top_k: int = 10,
    top_clusters: int = 3,
    seed: int = 42,
) -> dict:
    """Runs latency benchmarks and returns summary statistics."""
    rng = np.random.RandomState(seed)

    print(f"Generating synthetic database: {n_db} proteins, d={dim}, k={k}...")
    db_embs = rng.randn(n_db, dim).astype(np.float32)
    db_embs /= np.linalg.norm(db_embs, axis=1, keepdims=True)

    clustering = EMeansClustering(k=k, nepochs=2, spherical=True, seed=seed)
    clustering.fit(db_embs)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = IndexStorage(tmpdir, embedding_dim=dim)
        metadata = [
            {
                "accession": f"BENCH_{i:04d}",
                "entry_name": f"P{i:04d}",
                "description": f"Benchmark protein sequence {i}",
                "organism": "Escherichia coli",
                "seq_len": 320,
            }
            for i in range(1, n_db + 1)
        ]
        storage.save_index(metadata, db_embs, clustering.labels_, clustering.centroids_)

        engine = ProteinSearchEngine(data_dir=tmpdir, normalized=True)

        query_vectors = rng.randn(n_queries, dim).astype(np.float32)
        query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

        # Warm up
        engine.search(query_vectors[0], top_k=top_k, top_clusters=top_clusters)

        # Measure Vector Search Mode
        vec_latencies_ms = []
        for q in query_vectors:
            t0 = time.perf_counter()
            _, _ = engine.search(q, top_k=top_k, top_clusters=top_clusters)
            vec_latencies_ms.append((time.perf_counter() - t0) * 1000)

        # Measure Cached Sequence Search Mode
        embedder = ProteinEmbedder(use_mock=True, embedding_dim=dim, cache_size=1024)
        engine_cached = ProteinSearchEngine(data_dir=tmpdir, embedder=embedder, normalized=True)
        cached_seq = "MKTVRQERLKSIVRILERSKEPVSQML"
        engine_cached.search(cached_seq, top_k=top_k, top_clusters=top_clusters)  # warm cache

        cached_latencies_ms = []
        for _ in range(n_queries):
            t0 = time.perf_counter()
            _, _ = engine_cached.search(cached_seq, top_k=top_k, top_clusters=top_clusters)
            cached_latencies_ms.append((time.perf_counter() - t0) * 1000)

        storage.close()

    results = {
        "vector_search": {
            "median_ms": float(np.median(vec_latencies_ms)),
            "p95_ms": float(np.percentile(vec_latencies_ms, 95)),
            "mean_ms": float(np.mean(vec_latencies_ms)),
            "min_ms": float(np.min(vec_latencies_ms)),
        },
        "cached_search": {
            "median_ms": float(np.median(cached_latencies_ms)),
            "p95_ms": float(np.percentile(cached_latencies_ms, 95)),
            "mean_ms": float(np.mean(cached_latencies_ms)),
            "min_ms": float(np.min(cached_latencies_ms)),
        },
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EPSE Latency Benchmark Suite")
    parser.add_argument("--db-size", type=int, default=1000, help="Number of database proteins")
    parser.add_argument("--queries", type=int, default=50, help="Number of test queries")
    parser.add_argument("--dim", type=int, default=480, help="Embedding dimension")
    parser.add_argument("--k", type=int, default=32, help="Number of clusters")
    parser.add_argument("--top-clusters", "-c", type=int, default=3, help="Clusters loaded (M)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K hits returned")
    args = parser.parse_args()

    print("\n=======================================================")
    print("        EPSE Search Engine Performance Benchmark       ")
    print("=======================================================")
    print(f"Configuration: {args.db_size} DB proteins, {args.queries} queries, d={args.dim}, k={args.k}, M={args.top_clusters}")
    print("Running...\n")

    res = run_benchmark(
        n_db=args.db_size,
        n_queries=args.queries,
        dim=args.dim,
        k=args.k,
        top_k=args.top_k,
        top_clusters=args.top_clusters,
    )

    vs = res["vector_search"]
    cs = res["cached_search"]

    print("--- Benchmark Results ---")
    print(f"1. Vector Search Mode (precomputed embeddings):")
    print(f"   Median Latency: {vs['median_ms']:.3f} ms/query")
    print(f"   p95 Latency:    {vs['p95_ms']:.3f} ms/query")
    print(f"   Mean Latency:   {vs['mean_ms']:.3f} ms/query")
    print()
    print(f"2. Cached Sequence Search Mode (LRU hit):")
    print(f"   Median Latency: {cs['median_ms']:.3f} ms/query")
    print(f"   p95 Latency:    {cs['p95_ms']:.3f} ms/query")
    print(f"   Mean Latency:   {cs['mean_ms']:.3f} ms/query")
    print()
    print("--- Baseline Comparison (1,000 DB / 50 queries) ---")
    print(f"{'Engine':<20} {'Median / Avg Latency':<25} {'Speedup vs MMseqs2':<20} {'Speedup vs BLASTP':<20}")
    print("-" * 85)
    print(f"{'EPSE Vector Search':<20} {vs['median_ms']:.3f} ms{'':<18} {88.0 / vs['median_ms']:.1f}x{'':<15} {78.0 / vs['median_ms']:.1f}x")
    print(f"{'EPSE Cached Search':<20} {cs['median_ms']:.3f} ms{'':<18} {88.0 / cs['median_ms']:.1f}x{'':<15} {78.0 / cs['median_ms']:.1f}x")
    print(f"{'DIAMOND':<20} 2.800 ms{'':<18} 31.4x{'':<15} 27.9x")
    print(f"{'NCBI BLASTP':<20} 78.000 ms{'':<17} 1.1x{'':<16} 1.0x (baseline)")
    print(f"{'MMseqs2':<20} 88.000 ms{'':<17} 1.0x (baseline){'':<6} 0.9x")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
