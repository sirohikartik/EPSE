"""Tests for two-stage selective-load protein search engine."""

import numpy as np
import pytest
from epse.embedder import ProteinEmbedder
from epse.storage import IndexStorage
from epse.search import ProteinSearchEngine


def test_search_pipeline_integration(tmp_path):
    data_dir = str(tmp_path / "search_index")
    storage = IndexStorage(data_dir=data_dir, embedding_dim=16)

    # 10 proteins partitioned into 3 clusters
    metadata = [
        {
            "accession": f"P{i:03d}",
            "entry_name": f"PROT_{i}",
            "description": f"Enzyme family member {i}",
            "organism": "Homo sapiens" if i % 2 == 0 else "Mus musculus",
            "seq_len": 100 + i * 5,
        }
        for i in range(1, 11)
    ]

    # Cluster 0: indices 0..3 (proteins 1..4) - direction [1, 0, 0, ...]
    # Cluster 1: indices 4..6 (proteins 5..7) - direction [0, 1, 0, ...]
    # Cluster 2: indices 7..9 (proteins 8..10) - direction [0, 0, 1, ...]
    embeddings = np.zeros((10, 16), dtype=np.float32)
    for i in range(4):
        embeddings[i, 0] = 0.9 + 0.02 * i
        embeddings[i, 1] = 0.1
    for i in range(4, 7):
        embeddings[i, 1] = 0.9 + 0.02 * (i - 4)
        embeddings[i, 2] = 0.1
    for i in range(7, 10):
        embeddings[i, 2] = 0.9 + 0.02 * (i - 7)
        embeddings[i, 0] = 0.1

    # Normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    assignments = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    centroids = np.zeros((3, 16), dtype=np.float32)
    centroids[0] = np.mean(embeddings[0:4], axis=0)
    centroids[1] = np.mean(embeddings[4:7], axis=0)
    centroids[2] = np.mean(embeddings[7:10], axis=0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    storage.save_index(metadata, embeddings, assignments, centroids)

    mock_embedder = ProteinEmbedder(use_mock=True, embedding_dim=16)
    engine = ProteinSearchEngine(data_dir=data_dir, embedder=mock_embedder)

    assert engine.total_proteins == 10
    assert engine.k == 3

    # Query with a vector aligned with Cluster 0
    query_vec = np.zeros(16, dtype=np.float32)
    query_vec[0] = 1.0

    # Search top 2 candidates, loading only top 1 cluster
    hits, metrics = engine.search(query=query_vec, top_k=2, top_clusters=1)

    assert len(hits) == 2
    assert hits[0].accession in ["P001", "P002", "P003", "P004"]
    assert hits[0].cluster_id == 0
    assert hits[0].score > 0.8
    assert hits[0].organism in ["Homo sapiens", "Mus musculus"]

    # Verify selective loading metrics
    assert metrics.candidates_loaded == 4  # Only 4 proteins loaded out of 10
    assert metrics.clusters_loaded == 1
    assert metrics.search_space_pruned_pct == 60.0  # (1 - 4/10) * 100%
