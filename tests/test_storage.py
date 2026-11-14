"""Tests for EPSE disk storage and selective cluster reading."""

import os
import numpy as np
import pytest
from epse.storage import IndexStorage


def test_storage_save_and_selective_load(tmp_path):
    storage = IndexStorage(data_dir=str(tmp_path / "index"), embedding_dim=8)

    # 5 proteins, 2 clusters
    metadata = [
        {"accession": f"PROT_{i}", "entry_name": f"P{i}", "description": f"Desc {i}", "organism": "Human", "seq_len": 50 + i}
        for i in range(1, 6)
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
    ], dtype=np.float32)
    # Clusters: proteins 1, 2, 3 -> cluster 0; proteins 4, 5 -> cluster 1
    assignments = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    centroids = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.95, 0.05, 0.0, 0.0],
    ], dtype=np.float32)

    storage.save_index(metadata, embeddings, assignments, centroids)

    # Verify files created
    assert os.path.exists(storage.proteins_csv)
    assert os.path.exists(storage.embeddings_bin)
    assert os.path.exists(storage.assignments_bin)
    assert os.path.exists(storage.centroids_csv)
    assert os.path.exists(storage.cluster_index_path)

    # Test loading centroids
    loaded_centroids, counts = storage.load_centroids()
    assert loaded_centroids.shape == (2, 8)
    assert list(counts) == [3, 2]

    # Test selective cluster reading (ONLY cluster 1)
    pids, cl_embs = storage.load_cluster_embeddings([1])
    assert pids == [4, 5]
    assert cl_embs.shape == (2, 8)
    np.testing.assert_allclose(cl_embs[0], embeddings[3])
    np.testing.assert_allclose(cl_embs[1], embeddings[4])

    # Test metadata loading
    meta_dict = storage.load_protein_metadata([4])
    assert 4 in meta_dict
    assert meta_dict[4].accession == "PROT_4"
    assert meta_dict[4].organism == "Human"

    # Verify memory mapping handles close and clean reload
    assert storage._mmap is not None
    storage.close()
    assert storage._mmap is None
    pids_reloaded, _ = storage.load_cluster_embeddings([0])
    assert pids_reloaded == [1, 2, 3]
    storage.close()
