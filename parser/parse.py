"""EPSE Parser and Clustering Pipeline (Backward-Compatible Wrapper).

Upgraded to use modern E-Means clustering and universal FASTA parsing on CPU.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Union
import numpy as np
import torch

from epse.fasta import parse_fasta as epse_parse_fasta, UniversalFastaEntry
from epse.embedder import ProteinEmbedder, DEFAULT_MODEL_NAME, DEFAULT_EMBEDDING_DIM
from epse.emeans import EMeansClustering
from epse.storage import IndexStorage

# Configuration defaults
MODEL_NAME = DEFAULT_MODEL_NAME
DIM = DEFAULT_EMBEDDING_DIM
DEFAULT_THRESHOLD = 0.90
DEFAULT_FASTA = "uniprot_sprot.fasta"
BATCH_SIZE = 8


def parse_fasta(filepath: str):
    """Yields parsed protein entries from any protein database FASTA file.

    Maintains dictionary-like interface for backward compatibility.
    """
    for entry in epse_parse_fasta(filepath):
        yield entry.to_dict()


def cosine_similarity(centroid_vec: np.ndarray, embedding_vec: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    dot_product = float(np.dot(centroid_vec, embedding_vec))
    norm_c = float(np.linalg.norm(centroid_vec))
    norm_e = float(np.linalg.norm(embedding_vec))
    if norm_c == 0.0 or norm_e == 0.0:
        return 0.0
    return dot_product / (norm_c * norm_e)


def update_centroid(centroid_data: dict, embedding_vec: np.ndarray) -> None:
    """Legacy helper: updates a centroid dictionary with an incremental embedding."""
    centroid_data["count"] += 1
    centroid_data["vec"] += (embedding_vec - centroid_data["vec"]) / centroid_data["count"]


def get_embeddings_batch(model, tokenizer, sequences, device="cpu"):
    """Computes batch embeddings on CPU using ESM-2."""
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(torch.device("cpu"))

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            inputs["attention_mask"],
        )
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        elif isinstance(outputs, dict):
            hidden_states = outputs.get("last_hidden_state")
        else:
            hidden_states = outputs

        # Mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).expand_as(hidden_states).float()
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Normalize
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    return normalized.cpu().numpy()


def parse_and_save(
    fasta_path: str,
    threshold: Union[float, int] = DEFAULT_THRESHOLD,
    data_dir: str = ".data",
    use_mock: bool = False,
    nepochs: int = 5,
) -> None:
    """Parses arbitrary FASTA database, embeds on CPU, and clusters using E-Means."""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File {fasta_path} not found.")

    os.makedirs(data_dir, exist_ok=True)

    print(f"Loading FASTA entries from {fasta_path}...")
    entries = list(epse_parse_fasta(fasta_path))
    if not entries:
        print("No valid protein entries found.")
        return

    n_proteins = len(entries)
    print(f"Parsed {n_proteins} protein entries.")

    # Determine k from threshold or count
    if isinstance(threshold, int) and threshold >= 1:
        k = threshold
    elif isinstance(threshold, float) and threshold >= 1.0:
        k = int(threshold)
    else:
        # Map threshold/dataset size to reasonable number of clusters
        k = max(1, min(n_proteins, int(round(np.sqrt(n_proteins)))))

    print(f"Generating embeddings on CPU for {n_proteins} proteins...")
    embedder = ProteinEmbedder(
        model_name=MODEL_NAME,
        normalize=True,
        use_mock=use_mock,
        embedding_dim=DIM,
    )
    sequences = [e.seq for e in entries]
    embeddings = embedder.embed_sequences(sequences, batch_size=BATCH_SIZE)

    print(f"Running E-Means clustering on CPU (k={k}, epochs={nepochs})...")
    emeans = EMeansClustering(
        k=k,
        nepochs=nepochs,
        spherical=True,
        seed=1234,
    )
    emeans.fit(embeddings)

    print(f"Saving database artifacts to {data_dir}...")
    storage = IndexStorage(data_dir=data_dir, embedding_dim=embeddings.shape[1])
    metadata_list = [e.to_dict() for e in entries]
    storage.save_index(
        metadata_list=metadata_list,
        embeddings=embeddings,
        cluster_assignments=emeans.labels_,
        centroids=emeans.centroids_,
        cluster_counts=emeans.cluster_sizes_,
    )
    print("Done. Files written: proteins.csv, embeddings.bin, assignments.bin, centroids.csv, cluster_index.json")


def main():
    threshold = DEFAULT_THRESHOLD
    fasta_path = DEFAULT_FASTA

    if len(sys.argv) >= 2:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print("ERROR: threshold must be a number.", file=sys.stderr)
            sys.exit(1)

    if len(sys.argv) >= 3:
        fasta_path = sys.argv[2]

    print("=== EPSE Clustering (CPU E-Means) ===")
    print(f"Parameter : {threshold}")
    print(f"Dataset   : {fasta_path}\n")

    data_dir = f".data/threshold_{threshold:.2f}"
    parse_and_save(fasta_path, threshold, data_dir)


if __name__ == "__main__":
    main()
