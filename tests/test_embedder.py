"""Tests for CPU protein embedder."""

import numpy as np
import pytest
from epse.embedder import ProteinEmbedder


def test_mock_embedder_shapes_and_normalization():
    embedder = ProteinEmbedder(use_mock=True, embedding_dim=128, normalize=True)
    seqs = ["MKTVRQERLK", "MALWMRLLPLLALLALWGPDPAAA"]
    embs = embedder.embed_sequences(seqs)

    assert embs.shape == (2, 128)
    assert embs.dtype == np.float32

    # Verify L2 normalization
    for i in range(len(seqs)):
        norm = np.linalg.norm(embs[i])
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_embed_single():
    embedder = ProteinEmbedder(use_mock=True, embedding_dim=64)
    emb = embedder.embed_single("MKTVRQERLK")

    assert emb.shape == (64,)
    assert isinstance(emb, np.ndarray)


def test_empty_sequences():
    embedder = ProteinEmbedder(use_mock=True, embedding_dim=64)
    embs = embedder.embed_sequences([])
    assert embs.shape == (0, 64)


def test_embedder_lru_cache():
    embedder = ProteinEmbedder(use_mock=True, embedding_dim=32, cache_size=2)
    assert embedder.cache_size_current == 0

    emb1 = embedder.embed_single("SEQ1")
    assert embedder.cache_size_current == 1

    # Repeat should hit cache and be identical
    emb1_cached = embedder.embed_single("SEQ1")
    np.testing.assert_array_equal(emb1, emb1_cached)
    assert embedder.cache_size_current == 1

    # Fill cache to capacity
    emb2 = embedder.embed_single("SEQ2")
    assert embedder.cache_size_current == 2

    # Eviction test: adding third should evict oldest (SEQ1)
    emb3 = embedder.embed_single("SEQ3")
    assert embedder.cache_size_current == 2
    assert "SEQ1" not in embedder._cache
    assert "SEQ2" in embedder._cache
    assert "SEQ3" in embedder._cache

    # Clear cache
    embedder.clear_cache()
    assert embedder.cache_size_current == 0
