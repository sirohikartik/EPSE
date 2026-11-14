"""Modern E-Means clustering algorithm for high-dimensional embeddings.

Implements Meta FAIR's E-Means algorithm (arXiv:2611.09501, Nov 2026).
Provides:
1. Fast Faiss NumpyEmeans integration when faiss is available.
2. Self-contained pure-NumPy CPU reference implementation featuring:
   - Mini-batch Exponential Moving Average (EMA) updates
   - Empirical scaling laws for optimal batch size B and initial step alpha0
   - Cosine schedule for alpha annealing
   - Dead cluster revival via mass-proportional donor splitting
   - Finalization pass for exact centroid alignment
   - Spherical (cosine similarity) and Euclidean distance modes
"""

from __future__ import annotations

import os
import logging
import math
from typing import Optional, Tuple, Union
import numpy as np

# Prevent OpenMP duplicate initialization crash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


def compute_scaling_laws(
    n: int,
    k: int,
    epochs: float = 5.0,
    batch_law: str = "throughput",
) -> Tuple[int, float]:
    """Computes optimal batch size B and initial step size alpha_0 using e-means scaling laws.

    Formulas from arXiv:2611.09501:
    - Throughput law: B = 0.035 * k^0.13 * n^0.88 * T^0.81
    - Quality law: B_quality = B_throughput / 8
    - Step law: alpha_0 = min(0.824 * k^-0.46 * B^0.90 * n^-0.45 * T^-0.11, 1.0)
    """
    if n <= 0 or k <= 0:
        return 1, 1.0

    t = max(1.0, float(epochs))
    # Batch law
    raw_b = 0.035 * (k**0.13) * (n**0.88) * (t**0.81)
    if batch_law == "quality":
        raw_b /= 8.0

    b = int(np.clip(round(raw_b), 1, n))

    # Step law
    raw_alpha = 0.824 * (k**-0.46) * (b**0.90) * (n**-0.45) * (t**-0.11)
    alpha_0 = float(np.clip(raw_alpha, 0.001, 1.0))

    return b, alpha_0


class EMeansClustering:
    """E-Means clustering engine optimized for CPU vector embeddings."""

    def __init__(
        self,
        k: Optional[int] = None,
        nepochs: int = 5,
        spherical: bool = True,
        batch_size: Optional[int] = None,
        alpha0: Optional[int] = None,
        batch_law: str = "throughput",
        seed: int = 1234,
        prefer_faiss: bool = False,
        unused_threshold: float = 0.1,
    ):
        """Initializes E-Means clustering.

        Args:
            k: Number of clusters (if None, determined adaptively from n).
            nepochs: Epoch budget T (paper recommends 2 to 5 for matching 25-epoch Lloyd's).
            spherical: Whether to use spherical clustering (projected directions / cosine similarity).
            batch_size: Explicit batch size override (if None, set via scaling laws).
            alpha0: Explicit initial step size override (if None, set via scaling laws).
            batch_law: 'throughput' or 'quality' scaling law.
            seed: Random seed for initialization.
            prefer_faiss: If True, attempts to use faiss.contrib.e_means.NumpyEmeans if supported by system runtime.
            unused_threshold: Threshold fraction of mean mass below which dead clusters are revived.
        """
        self.k = k
        self.nepochs = max(1, nepochs)
        self.spherical = spherical
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.batch_law = batch_law
        self.seed = seed
        self.prefer_faiss = prefer_faiss
        self.unused_threshold = unused_threshold

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_sizes_: Optional[np.ndarray] = None

    def _auto_k(self, n: int) -> int:
        """Determines an appropriate number of clusters k based on dataset size."""
        if self.k is not None:
            return max(1, min(self.k, n))
        # Square root rule capped between 2 and 1000
        auto = int(round(math.sqrt(n)))
        return max(2, min(n, auto))

    def fit(self, X: np.ndarray, assume_normalized: bool = False) -> "EMeansClustering":
        """Fit on embeddings, optionally preserving an already-normalized memmap.

        ``assume_normalized`` avoids a full-array normalization copy when the
        caller's vectors are known to be L2-normalized, as ProteinEmbedder does.
        """
        X = np.asanyarray(X, dtype=np.float32)
        if not X.flags.c_contiguous:
            X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape

        if n == 0:
            raise ValueError("Input array X is empty.")

        k = self._auto_k(n)
        self.k = k

        if self.spherical and not assume_normalized:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms

        if k == 1:
            mean = np.mean(X, axis=0, keepdims=True)
            if self.spherical:
                norm = np.linalg.norm(mean)
                if norm > 0:
                    mean /= norm
            self.centroids_ = mean.astype(np.float32)
            self.labels_ = np.zeros(n, dtype=np.int32)
            self.cluster_sizes_ = np.array([n], dtype=np.int32)
            return self

        # Try Faiss NumpyEmeans first if preferred and available
        if self.prefer_faiss:
            try:
                from faiss.contrib.e_means import NumpyEmeans

                logger.info(
                    "Fitting e-means using faiss.contrib.e_means.NumpyEmeans (n=%d, k=%d, d=%d, T=%d)...",
                    n,
                    k,
                    d,
                    self.nepochs,
                )
                model = NumpyEmeans(
                    d=d,
                    k=k,
                    nepochs=self.nepochs,
                    batch_size=self.batch_size,
                    alpha0=self.alpha0,
                    batch_law=self.batch_law,
                    seed=self.seed,
                    spherical=self.spherical,
                    capture_assignments=True,
                )
                centroids, labels = model.train(X)
                self.centroids_ = centroids.astype(np.float32)
                # Compute exact final labels
                self.labels_ = self._assign(X, self.centroids_)
                self.cluster_sizes_ = np.bincount(self.labels_, minlength=k).astype(np.int32)
                return self
            except Exception as e:
                logger.warning("Faiss NumpyEmeans failed or unavailable (%s), using pure-NumPy E-Means.", e)

        # Standalone pure-NumPy E-Means implementation
        self._fit_numpy(X, n, d, k)
        return self

    def _fit_numpy(self, X: np.ndarray, n: int, d: int, k: int) -> None:
        """Self-contained implementation of E-Means algorithm with EMA and scaling laws."""
        logger.info(
            "Fitting e-means via pure-NumPy implementation (n=%d, k=%d, d=%d, T=%d)...",
            n,
            k,
            d,
            self.nepochs,
        )
        rng = np.random.RandomState(self.seed)

        # 1. Scaling laws
        if self.batch_size is not None and self.alpha0 is not None:
            batch_size = min(n, max(1, self.batch_size))
            alpha_0 = self.alpha0
        else:
            law_b, law_alpha = compute_scaling_laws(n, k, epochs=self.nepochs, batch_law=self.batch_law)
            batch_size = self.batch_size if self.batch_size is not None else law_b
            alpha_0 = self.alpha0 if self.alpha0 is not None else law_alpha

        # 2. Initialization: random selection of k distinct points
        init_indices = rng.choice(n, size=k, replace=False)
        centroids = X[init_indices].astype(np.float64).copy()

        if self.spherical:
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            c_norms[c_norms == 0] = 1.0
            centroids /= c_norms

        # S stores accumulated vector sums; N stores mass
        S = centroids.copy()
        N = np.full((k,), 2.0**-16, dtype=np.float64)

        steps_per_epoch = int(math.ceil(n / batch_size))
        total_steps = steps_per_epoch * self.nepochs
        step_idx = 0

        for epoch in range(self.nepochs):
            perm = rng.permutation(n)

            for b_start in range(0, n, batch_size):
                b_indices = perm[b_start : b_start + batch_size]
                batch_X = X[b_indices]
                b_len = len(batch_X)

                # Cosine schedule for alpha decay
                progress = float(step_idx) / max(1, total_steps - 1)
                alpha = alpha_0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

                # Nearest centroid assignment
                if self.spherical:
                    # Dot product for unit-normalized vectors
                    sims = np.dot(batch_X, centroids.T)  # (b_len, k)
                    assignments = np.argmax(sims, axis=1)
                else:
                    # Euclidean distance squared: ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
                    x_sq = np.sum(batch_X**2, axis=1, keepdims=True)
                    c_sq = np.sum(centroids**2, axis=1, keepdims=True).T
                    dists = x_sq - 2.0 * np.dot(batch_X, centroids.T) + c_sq
                    assignments = np.argmin(dists, axis=1)

                # Accumulate batch counts and sums
                s_batch = np.zeros_like(centroids)
                n_batch = np.zeros(k, dtype=np.float64)

                for cid in range(k):
                    mask = assignments == cid
                    if np.any(mask):
                        s_batch[cid] = np.sum(batch_X[mask], axis=0)
                        n_batch[cid] = float(np.sum(mask))

                # Exponential Moving Average update
                S = (1.0 - alpha) * S + alpha * s_batch
                N = (1.0 - alpha) * N + alpha * n_batch

                # Update centroid positions c_j = S_j / N_j
                valid_mask = N > 0
                centroids[valid_mask] = S[valid_mask] / N[valid_mask, np.newaxis]

                # Dead cluster revival every 5 steps
                if (step_idx + 1) % 5 == 0 and d > 2:
                    mean_mass = np.mean(N)
                    dead = np.where(N < self.unused_threshold * mean_mass)[0]
                    if len(dead) > 0 and len(dead) < k:
                        alive = np.where(N >= self.unused_threshold * mean_mass)[0]
                        if len(alive) > 0:
                            probs = N[alive] / np.sum(N[alive])
                            donors = rng.choice(alive, size=len(dead), replace=True, p=probs)
                            for dead_id, donor_id in zip(dead, donors):
                                # Perturb donor centroid slightly to revive dead cluster
                                noise = rng.randn(d) * 1e-4
                                centroids[dead_id] = centroids[donor_id] + noise
                                N[dead_id] = N[donor_id] * 0.5
                                N[donor_id] = N[donor_id] * 0.5
                                S[dead_id] = centroids[dead_id] * N[dead_id]
                                S[donor_id] = centroids[donor_id] * N[donor_id]

                if self.spherical:
                    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                    c_norms[c_norms == 0] = 1.0
                    centroids /= c_norms

                step_idx += 1

        # 3. Finalization step: Exact mean calculation over final assignments
        final_labels = self._assign(X, centroids)
        sums = np.zeros((k, d), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        chunk_size = 8192
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            chunk_labels = final_labels[start:end]
            chunk_X = X[start:end]
            for cid in np.unique(chunk_labels):
                mask = chunk_labels == cid
                sums[cid] += np.sum(chunk_X[mask], axis=0)
                counts[cid] += int(np.sum(mask))

        final_centroids = centroids.copy()
        nonempty = counts > 0
        final_centroids[nonempty] = sums[nonempty] / counts[nonempty, np.newaxis]

        if self.spherical:
            fc_norms = np.linalg.norm(final_centroids, axis=1, keepdims=True)
            fc_norms[fc_norms == 0] = 1.0
            final_centroids /= fc_norms

        self.centroids_ = final_centroids.astype(np.float32)
        self.labels_ = self._assign(X, self.centroids_)
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=k).astype(np.int32)

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assigns rows of X to the nearest centroid."""
        labels = np.empty(len(X), dtype=np.int32)
        chunk_size = 8192
        c_sq = np.sum(centroids**2, axis=1, keepdims=True).T
        for start in range(0, len(X), chunk_size):
            end = min(len(X), start + chunk_size)
            chunk = X[start:end]
            if self.spherical:
                scores = np.dot(chunk, centroids.T)
                labels[start:end] = np.argmax(scores, axis=1)
            else:
                x_sq = np.sum(chunk**2, axis=1, keepdims=True)
                dists = x_sq - 2.0 * np.dot(chunk, centroids.T) + c_sq
                labels[start:end] = np.argmin(dists, axis=1)
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts closest cluster indices for given embeddings."""
        if self.centroids_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.spherical:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return self._assign(X, self.centroids_)

    def score_query(self, query_vec: np.ndarray, top_m: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Ranks cluster centroids by similarity to query_vec.

        Args:
            query_vec: 1D array of shape (d,) representing query embedding.
            top_m: Number of top clusters to retrieve.

        Returns:
            Tuple of (top_cluster_indices, similarity_scores).
        """
        if self.centroids_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        q = query_vec.astype(np.float32).flatten()
        if self.spherical:
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            scores = np.dot(self.centroids_, q)
            # Higher dot product is better
            ranked_indices = np.argsort(-scores)[:top_m]
            return ranked_indices, scores[ranked_indices]
        else:
            dists = np.linalg.norm(self.centroids_ - q, axis=1)
            # Smaller distance is better
            ranked_indices = np.argsort(dists)[:top_m]
            return ranked_indices, dists[ranked_indices]
