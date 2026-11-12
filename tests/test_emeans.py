"""Tests for the E-Means clustering engine."""

import numpy as np
import pytest
from epse.emeans import EMeansClustering, compute_scaling_laws


def test_scaling_laws_computation():
    n = 10000
    k = 50
    epochs = 5.0
    b, alpha_0 = compute_scaling_laws(n, k, epochs=epochs, batch_law="throughput")

    assert 1 <= b <= n
    assert 0.0 < alpha_0 <= 1.0

    b_qual, _ = compute_scaling_laws(n, k, epochs=epochs, batch_law="quality")
    assert b_qual < b


def test_emeans_pure_numpy_spherical():
    # Generate synthetic clustered unit vectors
    rng = np.random.RandomState(42)
    n_per_cluster = 30
    k_true = 3
    d = 16

    true_centers = rng.randn(k_true, d).astype(np.float32)
    true_centers /= np.linalg.norm(true_centers, axis=1, keepdims=True)

    points = []
    for center in true_centers:
        cluster_pts = center + 0.1 * rng.randn(n_per_cluster, d).astype(np.float32)
        cluster_pts /= np.linalg.norm(cluster_pts, axis=1, keepdims=True)
        points.append(cluster_pts)

    X = np.vstack(points)

    # Force pure-numpy fallback to test algorithm implementation
    emeans = EMeansClustering(k=k_true, nepochs=5, spherical=True, prefer_faiss=False, seed=42)
    emeans.fit(X)

    assert emeans.centroids_ is not None
    assert emeans.centroids_.shape == (k_true, d)
    assert len(emeans.labels_) == len(X)
    assert len(emeans.cluster_sizes_) == k_true

    # Centroids should be unit vectors
    norms = np.linalg.norm(emeans.centroids_, axis=1)
    for norm in norms:
        assert pytest.approx(norm, abs=1e-4) == 1.0


def test_emeans_euclidean_and_auto_k():
    rng = np.random.RandomState(42)
    X = rng.randn(64, 16).astype(np.float32)

    # Test auto k selection and Euclidean distance
    emeans = EMeansClustering(k=None, nepochs=4, spherical=False, seed=42)
    emeans.fit(X)

    # auto k for n=64 is round(sqrt(64)) = 8
    assert emeans.k == 8
    assert emeans.centroids_.shape == (8, 16)
    assert len(emeans.labels_) == 64
    assert np.all(emeans.labels_ >= 0) and np.all(emeans.labels_ < 8)


def test_emeans_query_scoring():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 16).astype(np.float32)

    emeans = EMeansClustering(k=4, nepochs=3, spherical=True, seed=42)
    emeans.fit(X)

    query = X[0]
    top_clusters, scores = emeans.score_query(query, top_m=2)

    assert len(top_clusters) == 2
    assert len(scores) == 2
    assert scores[0] >= scores[1]  # Ranked descending
