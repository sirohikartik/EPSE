"""Disk-backed storage and inverted cluster index for EPSE.

Provides constant-time O(1) random disk access to protein embeddings and metadata,
allowing the search engine to selectively load only top-scoring clusters into memory.
"""

from __future__ import annotations

import csv
import json
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ProteinMetadata:
    """Metadata for a single protein record."""
    protein_id: int
    accession: str
    entry_name: str
    description: str
    organism: str
    seq_len: int


class IndexStorage:
    """Manages reading and writing of EPSE database files."""

    def __init__(self, data_dir: str, embedding_dim: int = 480):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.proteins_csv = os.path.join(data_dir, "proteins.csv")
        self.embeddings_bin = os.path.join(data_dir, "embeddings.bin")
        self.assignments_bin = os.path.join(data_dir, "assignments.bin")
        self.centroids_csv = os.path.join(data_dir, "centroids.csv")
        self.cluster_index_path = os.path.join(data_dir, "cluster_index.json")

        # Record size in embeddings.bin: 4 bytes (uint32 id) + 4 * embedding_dim (float32s)
        self.record_size = 4 + 4 * self.embedding_dim

    def save_index(
        self,
        metadata_list: List[Dict[str, Union[str, int]]],
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        centroids: np.ndarray,
        cluster_counts: Optional[np.ndarray] = None,
    ) -> None:
        """Saves full index files to disk in standard EPSE format."""
        os.makedirs(self.data_dir, exist_ok=True)
        n = len(metadata_list)
        dim = embeddings.shape[1]
        self.embedding_dim = dim
        self.record_size = 4 + 4 * dim

        # 1. Write proteins.csv
        with open(self.proteins_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "accession", "entry_name", "description", "organism", "seq_len"])
            for idx, meta in enumerate(metadata_list, start=1):
                writer.writerow([
                    idx,
                    meta.get("accession", "unknown"),
                    meta.get("entry_name", "unknown"),
                    meta.get("description", ""),
                    meta.get("organism", "unknown"),
                    meta.get("seq_len", 0),
                ])

        # 2. Write embeddings.bin (uint32 protein_id + float32 array)
        with open(self.embeddings_bin, "wb") as f:
            for idx in range(1, n + 1):
                f.write(struct.pack("I", idx))
                f.write(embeddings[idx - 1].astype(np.float32).tobytes())

        # 3. Write assignments.bin (uint32 protein_id + uint32 cluster_id)
        cluster_to_pids: Dict[int, List[int]] = {}
        with open(self.assignments_bin, "wb") as f:
            for idx in range(1, n + 1):
                cid = int(cluster_assignments[idx - 1])
                f.write(struct.pack("II", idx, cid))
                cluster_to_pids.setdefault(cid, []).append(idx)

        # 4. Write centroids.csv
        k = len(centroids)
        if cluster_counts is None:
            cluster_counts = np.array([len(cluster_to_pids.get(c, [])) for c in range(k)], dtype=np.int32)

        with open(self.centroids_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["centroid_id", "count"] + [f"e{i}" for i in range(dim)]
            writer.writerow(header)
            for cid in range(k):
                row = [cid, int(cluster_counts[cid])] + centroids[cid].tolist()
                writer.writerow(row)

        # 5. Write cluster_index.json (inverted index for fast cluster lookup)
        with open(self.cluster_index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_dim": dim,
                    "num_proteins": n,
                    "num_clusters": k,
                    "clusters": {str(k): v for k, v in cluster_to_pids.items()},
                },
                f,
                indent=2,
            )

    def load_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads cluster centroids and counts from centroids.csv.

        Returns:
            Tuple of (centroids array of shape (k, d), counts array of shape (k,)).
        """
        if not os.path.exists(self.centroids_csv):
            raise FileNotFoundError(f"Centroids file not found at: {self.centroids_csv}")

        centroid_list = []
        counts_list = []

        with open(self.centroids_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.embedding_dim = len(header) - 2
            self.record_size = 4 + 4 * self.embedding_dim

            for row in reader:
                if not row:
                    continue
                cid = int(row[0])
                cnt = int(row[1])
                vec = [float(x) for x in row[2:]]
                counts_list.append(cnt)
                centroid_list.append(vec)

        return np.array(centroid_list, dtype=np.float32), np.array(counts_list, dtype=np.int32)

    def load_cluster_index(self) -> Dict[int, List[int]]:
        """Loads inverted index mapping cluster_id -> list of protein_ids."""
        if os.path.exists(self.cluster_index_path):
            with open(self.cluster_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.get("clusters", {}).items()}

        # Fallback: rebuild from assignments.bin if json doesn't exist
        if not os.path.exists(self.assignments_bin):
            raise FileNotFoundError(f"Assignments file not found at: {self.assignments_bin}")

        cluster_to_pids: Dict[int, List[int]] = {}
        with open(self.assignments_bin, "rb") as f:
            chunk = f.read(8)
            while chunk and len(chunk) == 8:
                pid, cid = struct.unpack("II", chunk)
                cluster_to_pids.setdefault(cid, []).append(pid)
                chunk = f.read(8)

        return cluster_to_pids

    def load_cluster_embeddings(
        self,
        cluster_ids: List[int],
        cluster_index: Optional[Dict[int, List[int]]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """Selectively reads embeddings ONLY for specified cluster IDs directly from disk.

        This ensures minimal RAM consumption: non-matching clusters are never loaded.

        Returns:
            Tuple of (protein_ids, embeddings matrix of shape (num_selected, dim)).
        """
        if cluster_index is None:
            cluster_index = self.load_cluster_index()

        target_pids: List[int] = []
        for cid in cluster_ids:
            target_pids.extend(cluster_index.get(cid, []))

        if not target_pids:
            return [], np.empty((0, self.embedding_dim), dtype=np.float32)

        # Sort pids so nearby records can be fetched in one sequential read.
        sorted_pids = sorted(set(target_pids))
        embeddings = np.zeros((len(sorted_pids), self.embedding_dim), dtype=np.float32)
        record_dtype = np.dtype(
            [("protein_id", np.uint32), ("embedding", np.float32, (self.embedding_dim,))]
        )
        max_gap_records = 8

        with open(self.embeddings_bin, "rb") as f:
            run_start = 0
            while run_start < len(sorted_pids):
                run_end = run_start + 1
                while (
                    run_end < len(sorted_pids)
                    and sorted_pids[run_end] - sorted_pids[run_end - 1] <= max_gap_records + 1
                ):
                    run_end += 1

                first_pid = sorted_pids[run_start]
                last_pid = sorted_pids[run_end - 1]
                offset = (first_pid - 1) * self.record_size
                f.seek(offset)
                raw = f.read((last_pid - first_pid + 1) * self.record_size)
                records = np.frombuffer(raw, dtype=record_dtype)

                for out_idx in range(run_start, run_end):
                    record_idx = sorted_pids[out_idx] - first_pid
                    if record_idx < len(records):
                        embeddings[out_idx] = records[record_idx]["embedding"]

                run_start = run_end

        return sorted_pids, embeddings

    def load_protein_metadata(self, protein_ids: Optional[List[int]] = None) -> Dict[int, ProteinMetadata]:
        """Loads metadata for specified protein IDs (or all if None)."""
        if not os.path.exists(self.proteins_csv):
            raise FileNotFoundError(f"Proteins CSV not found: {self.proteins_csv}")

        pid_set = set(protein_ids) if protein_ids is not None else None
        results: Dict[int, ProteinMetadata] = {}

        with open(self.proteins_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)  # Header
            for row in reader:
                if not row:
                    continue
                pid = int(row[0])
                if pid_set is not None and pid not in pid_set:
                    continue
                results[pid] = ProteinMetadata(
                    protein_id=pid,
                    accession=row[1],
                    entry_name=row[2],
                    description=row[3],
                    organism=row[4],
                    seq_len=int(row[5]),
                )

        return results
