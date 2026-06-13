#!/usr/bin/env python3
"""
Online centroid assignment with configurable similarity threshold.
"""

import argparse
import logging
import sys
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Online centroid assignment with configurable threshold"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input embeddings file (npy format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clusters.npy",
        help="Path to output clusters file (default: clusters.npy)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for centroid assignment (default: 0.90)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(args)


def online_centroid_assignment(
    embeddings: np.ndarray,
    threshold: float = 0.90,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Assign embeddings to centroids using online clustering.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        threshold: Similarity threshold for centroid assignment

    Returns:
        Tuple of (centroids, cluster_assignments)
        - centroids: List of centroid vectors
        - cluster_assignments: List of lists containing indices assigned to each centroid
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return [], []

    centroids: List[np.ndarray] = []
    cluster_assignments: List[List[int]] = []

    for i in range(n_samples):
        embedding = embeddings[i].reshape(1, -1)

        if len(centroids) == 0:
            # First embedding becomes first centroid
            centroids.append(embedding.flatten())
            cluster_assignments.append([i])
            continue

        # Compute similarity with all existing centroids
        centroid_matrix = np.array(centroids)
        similarities = cosine_similarity(embedding, centroid_matrix)[0]

        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]

        if max_sim >= threshold:
            # Assign to existing centroid
            cluster_assignments[max_sim_idx].append(i)
            # Update centroid (online mean update)
            cluster_size = len(cluster_assignments[max_sim_idx])
            centroids[max_sim_idx] = (
                centroids[max_sim_idx] * (cluster_size - 1) + embedding.flatten()
            ) / cluster_size
        else:
            # Create new centroid
            centroids.append(embedding.flatten())
            cluster_assignments.append([i])

    return centroids, cluster_assignments


def main(args: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parsed_args = parse_args(args)

    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading embeddings from %s", parsed_args.input)
    try:
        embeddings = np.load(parsed_args.input)
    except FileNotFoundError:
        logger.error("Input file not found: %s", parsed_args.input)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid input file: %s", e)
        sys.exit(1)

    logger.info(
        "Running online centroid assignment with threshold=%.2f", parsed_args.threshold
    )
    logger.info("Number of embeddings: %d", embeddings.shape[0])

    centroids, cluster_assignments = online_centroid_assignment(
        embeddings, threshold=parsed_args.threshold
    )

    logger.info("Number of clusters: %d", len(centroids))
    logger.info("Cluster sizes: %s", [len(c) for c in cluster_assignments])

    # Save cluster assignments
    output_data = {
        "centroids": np.array(centroids) if centroids else np.array([]),
        "assignments": cluster_assignments,
        "threshold": parsed_args.threshold,
    }
    np.save(parsed_args.output, output_data)
    logger.info("Results saved to %s", parsed_args.output)


if __name__ == "__main__":
    main()