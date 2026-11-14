"""Two-stage selective-load protein similarity search engine for EPSE.

Executes low-memory similarity queries over large protein datasets on CPU by:
1. Generating a query embedding.
2. Ranking cluster centroids using cosine similarity.
3. Loading only the top-M relevant clusters from disk into RAM.
4. Computing fine-grained similarity within the pruned candidate subset.
5. Returning ranked matches with detailed annotations and performance metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .embedder import ProteinEmbedder
from .storage import IndexStorage, ProteinMetadata

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a matched protein hit."""
    rank: int
    protein_id: int
    accession: str
    entry_name: str
    organism: str
    description: str
    score: float
    cluster_id: int
    seq_len: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "protein_id": self.protein_id,
            "accession": self.accession,
            "entry_name": self.entry_name,
            "organism": self.organism,
            "description": self.description,
            "score": round(float(self.score), 4),
            "cluster_id": self.cluster_id,
            "seq_len": self.seq_len,
        }


@dataclass
class SearchMetrics:
    """Performance and efficiency metrics for a search query."""
    total_proteins: int
    candidates_loaded: int
    clusters_evaluated: int
    clusters_loaded: int
    search_space_pruned_pct: float
    embed_time_sec: float
    centroid_search_time_sec: float
    disk_load_time_sec: float
    ranking_time_sec: float
    total_time_sec: float


class ProteinSearchEngine:
    """High-efficiency protein search engine operating on CPU."""

    def __init__(
        self,
        data_dir: str,
        embedder: Optional[ProteinEmbedder] = None,
    ):
        """Initializes the search engine from an indexed directory.

        Args:
            data_dir: Directory containing proteins.csv, embeddings.bin, centroids.csv, etc.
            embedder: Optional ProteinEmbedder instance. If None, initialized on CPU.
        """
        self.data_dir = data_dir
        self.storage = IndexStorage(data_dir)
        self.embedder = embedder

        # Load centroids and inverted cluster index
        self.centroids, self.cluster_counts = self.storage.load_centroids()
        self.cluster_index = self.storage.load_cluster_index()
        # Metadata is small compared with embeddings and is needed on every
        # response. Parse it once instead of rescanning the CSV per query.
        self._metadata = self.storage.load_protein_metadata()

        self.k, self.dim = self.centroids.shape
        self.storage.embedding_dim = self.dim
        self.storage.record_size = 4 + 4 * self.dim

        # Map each protein_id to its assigned cluster_id
        self._pid_to_cluster: Dict[int, int] = {}
        for cid, pids in self.cluster_index.items():
            for pid in pids:
                self._pid_to_cluster[pid] = cid

        self.total_proteins = sum(len(pids) for pids in self.cluster_index.values())
        logger.info(
            "Search engine ready: %d proteins partitioned into %d clusters (d=%d).",
            self.total_proteins,
            self.k,
            self.dim,
        )

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        top_clusters: int = 3,
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Performs selective-load similarity search for a query sequence or embedding.

        Args:
            query: Amino acid sequence string or precomputed 1D numpy embedding.
            top_k: Maximum number of ranked protein hits to return.
            top_clusters: Number of nearest clusters to selectively load from disk.

        Returns:
            Tuple of (list of SearchResult, SearchMetrics).
        """
        start_total = time.perf_counter()

        # Step 1: Query embedding
        t0 = time.perf_counter()
        if isinstance(query, str):
            if self.embedder is None:
                self.embedder = ProteinEmbedder(normalize=True)
            query_vec = self.embedder.embed_single(query)
        else:
            query_vec = query.astype(np.float32).flatten()
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
        t_embed = time.perf_counter() - t0

        # Step 2: Compare against cluster centroids using cosine similarity
        t0 = time.perf_counter()
        centroid_scores = np.dot(self.centroids, query_vec)
        # Select top-M clusters
        num_clusters_to_load = min(max(1, top_clusters), self.k)
        ranked_clusters = np.argsort(-centroid_scores)[:num_clusters_to_load].tolist()
        t_centroid = time.perf_counter() - t0

        # Step 3: Selectively load ONLY candidate embeddings from disk
        t0 = time.perf_counter()
        pids, candidate_embs = self.storage.load_cluster_embeddings(
            ranked_clusters, cluster_index=self.cluster_index
        )
        t_disk = time.perf_counter() - t0

        if len(pids) == 0:
            metrics = SearchMetrics(
                total_proteins=self.total_proteins,
                candidates_loaded=0,
                clusters_evaluated=self.k,
                clusters_loaded=len(ranked_clusters),
                search_space_pruned_pct=100.0,
                embed_time_sec=t_embed,
                centroid_search_time_sec=t_centroid,
                disk_load_time_sec=t_disk,
                ranking_time_sec=0.0,
                total_time_sec=time.perf_counter() - start_total,
            )
            return [], metrics

        # Step 4: Fine-grained cosine ranking within candidates
        t0 = time.perf_counter()
        # candidate_embs are unit-normalized if generated with normalize=True
        # Compute cosine similarity
        cand_norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1.0
        norm_embs = candidate_embs / cand_norms
        similarities = np.dot(norm_embs, query_vec)

        # Rank candidates
        rank_order = np.argsort(-similarities)[:top_k]
        top_pids = [pids[idx] for idx in rank_order]
        top_scores = [float(similarities[idx]) for idx in rank_order]
        t_rank = time.perf_counter() - t0

        # Load metadata only for the top-K returned hits
        results: List[SearchResult] = []
        for rank, (pid, score) in enumerate(zip(top_pids, top_scores), start=1):
            meta = self._metadata.get(pid)
            results.append(
                SearchResult(
                    rank=rank,
                    protein_id=pid,
                    accession=meta.accession if meta else f"ID_{pid}",
                    entry_name=meta.entry_name if meta else f"ID_{pid}",
                    organism=meta.organism if meta else "unknown",
                    description=meta.description if meta else "",
                    score=score,
                    cluster_id=self._pid_to_cluster.get(pid, -1),
                    seq_len=meta.seq_len if meta else 0,
                )
            )

        candidates_count = len(pids)
        pruned_pct = (
            100.0 * (1.0 - candidates_count / max(1, self.total_proteins))
            if self.total_proteins > 0
            else 0.0
        )

        metrics = SearchMetrics(
            total_proteins=self.total_proteins,
            candidates_loaded=candidates_count,
            clusters_evaluated=self.k,
            clusters_loaded=len(ranked_clusters),
            search_space_pruned_pct=pruned_pct,
            embed_time_sec=t_embed,
            centroid_search_time_sec=t_centroid,
            disk_load_time_sec=t_disk,
            ranking_time_sec=t_rank,
            total_time_sec=time.perf_counter() - start_total,
        )

        return results, metrics
