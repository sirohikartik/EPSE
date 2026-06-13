#!/usr/bin/env python3
"""
Evaluate clustering behavior across multiple similarity thresholds.

This script runs online centroid assignment on a SwissProt subset
with different thresholds and records clustering statistics to
determine the optimal threshold.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.epse.online_centroid_assignment import online_centroid_assignment

logger = logging.getLogger(__name__)

# Thresholds to evaluate
THRESHOLDS = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate similarity thresholds for centroid assignment"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to SwissProt embeddings file (npy format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="threshold_evaluation",
        help="Directory for output files (default: threshold_evaluation)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=THRESHOLDS,
        help="Thresholds to evaluate (default: 0.90 0.92 0.94 0.96 0.98 0.99)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def compute_cluster_statistics(
    cluster_assignments: List[List[int]],
) -> Dict[str, float]:
    """
    Compute statistics for a set of clusters.

    Args:
        cluster_assignments: List of lists containing indices assigned to each centroid

    Returns:
        Dictionary with statistics:
        - num_clusters: Number of clusters
        - largest_cluster_size: Size of the largest cluster
        - average_cluster_size: Average cluster size
        - median_cluster_size: Median cluster size
        - std_cluster_size: Standard deviation of cluster sizes
    """
    if not cluster_assignments:
        return {
            "num_clusters": 0,
            "largest_cluster_size": 0,
            "average_cluster_size": 0.0,
            "median_cluster_size": 0.0,
            "std_cluster_size": 0.0,
        }

    cluster_sizes = [len(c) for c in cluster_assignments]
    cluster_sizes.sort()

    return {
        "num_clusters": len(cluster_sizes),
        "largest_cluster_size": cluster_sizes[-1],
        "average_cluster_size": float(np.mean(cluster_sizes)),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "std_cluster_size": float(np.std(cluster_sizes)),
    }


def plot_cluster_size_histogram(
    cluster_sizes: List[int],
    threshold: float,
    output_dir: Path,
) -> None:
    """
    Plot and save histogram of cluster sizes.

    Args:
        cluster_sizes: List of cluster sizes
        threshold: Similarity threshold used
        output_dir: Directory to save the plot
    """
    if not cluster_sizes:
        logger.warning("No clusters to plot for threshold %.2f", threshold)
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use logarithmic scale for better visualization
    ax.hist(cluster_sizes, bins=50, alpha=0.7, edgecolor="black", log=True)
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(f"Cluster Size Distribution (Threshold = {threshold:.2f})")

    # Add statistics as text
    stats_text = (
        f"Total clusters: {len(cluster_sizes)}\n"
        f"Largest: {max(cluster_sizes)}\n"
        f"Average: {np.mean(cluster_sizes):.1f}\n"
        f"Median: {np.median(cluster_sizes):.1f}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    output_path = output_dir / f"histogram_threshold_{threshold:.2f}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Histogram saved to %s", output_path)


def select_optimal_threshold(
    results: Dict[float, Dict[str, float]],
) -> Tuple[float, str]:
    """
    Select the optimal threshold based on clustering statistics.

    The optimal threshold minimizes giant-cluster collapse while
    avoiding excessive fragmentation.

    Args:
        results: Dictionary mapping thresholds to their statistics

    Returns:
        Tuple of (optimal_threshold, explanation)
    """
    if not results:
        raise ValueError("No results to evaluate")

    # Score each threshold based on:
    # 1. Avoid giant clusters (penalize large largest_cluster_size)
    # 2. Avoid excessive fragmentation (penalize too many clusters)
    # 3. Balance cluster sizes (prefer lower std/mean ratio)

    scores = {}
    for threshold, stats in results.items():
        if stats["num_clusters"] == 0:
            scores[threshold] = float("-inf")
            continue

        # Normalize metrics
        num_clusters = stats["num_clusters"]
        largest_cluster = stats["largest_cluster_size"]
        avg_cluster = stats["average_cluster_size"]
        std_cluster = stats["std_cluster_size"]

        # Coefficient of variation (lower is better for balance)
        cv = std_cluster / avg_cluster if avg_cluster > 0 else float("inf")

        # Score: prefer balanced clusters without excessive fragmentation
        # Penalize: high CV, very large clusters, too many clusters
        score = -cv - (largest_cluster / avg_cluster) * 0.1 + (num_clusters**0.5) * 0.01

        scores[threshold] = score

    optimal_threshold = max(scores, key=scores.get)

    explanation = (
        f"Selected threshold {optimal_threshold:.2f} based on:\n"
        f"- Coefficient of variation: {results[optimal_threshold]['std_cluster_size'] / results[optimal_threshold]['average_cluster_size']:.2f}\n"
        f"- Number of clusters: {results[optimal_threshold]['num_clusters']}\n"
        f"- Largest cluster size: {results[optimal_threshold]['largest_cluster_size']}\n"
        f"- Average cluster size: {results[optimal_threshold]['average_cluster_size']:.1f}"
    )

    return optimal_threshold, explanation


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    logger.info("Loading embeddings from %s", args.input)
    try:
        embeddings = np.load(args.input)
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid input file: %s", e)
        sys.exit(1)

    logger.info("Number of embeddings: %d", embeddings.shape[0])
    logger.info("Embedding dimension: %d", embeddings.shape[1])

    # Evaluate each threshold
    results: Dict[float, Dict[str, float]] = {}

    for threshold in sorted(args.thresholds):
        logger.info("=" * 60)
        logger.info("Evaluating threshold: %.2f", threshold)

        try:
            centroids, cluster_assignments = online_centroid_assignment(
                embeddings, threshold=threshold
            )
        except Exception as e:
            logger.error("Error processing threshold %.2f: %s", threshold, e)
            continue

        stats = compute_cluster_statistics(cluster_assignments)
        results[threshold] = stats

        logger.info("Number of clusters: %d", stats["num_clusters"])
        logger.info("Largest cluster size: %d", stats["largest_cluster_size"])
        logger.info("Average cluster size: %.2f", stats["average_cluster_size"])
        logger.info("Median cluster size: %.2f", stats["median_cluster_size"])
        logger.info("Std cluster size: %.2f", stats["std_cluster_size"])

        # Plot histogram
        cluster_sizes = [len(c) for c in cluster_assignments]
        plot_cluster_size_histogram(cluster_sizes, threshold, output_dir)

    # Generate comparison table
    logger.info("=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)
    logger.info(
        "%-10s %-15s %-20s %-20s %-20s %-15s",
        "Threshold",
        "Num Clusters",
        "Largest Cluster",
        "Average Size",
        "Median Size",
        "Std Size",
    )
    logger.info("-" * 100)

    for threshold in sorted(results.keys()):
        stats = results[threshold]
        logger.info(
            "%-10.2f %-15d %-20d %-20.2f %-20.2f %-15.2f",
            threshold,
            stats["num_clusters"],
            stats["largest_cluster_size"],
            stats["average_cluster_size"],
            stats["median_cluster_size"],
            stats["std_cluster_size"],
        )

    # Select optimal threshold
    if results:
        optimal_threshold, explanation = select_optimal_threshold(results)
        logger.info("=" * 60)
        logger.info("OPTIMAL THRESHOLD: %.2f", optimal_threshold)
        logger.info(explanation)

        # Save results to JSON
        output_data = {
            "thresholds": {
                f"{t:.2f}": results[t] for t in sorted(results.keys())
            },
            "optimal_threshold": optimal_threshold,
            "explanation": explanation,
        }

        output_path = output_dir / "threshold_evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", output_path)
    else:
        logger.error("No valid results to evaluate")


if __name__ == "__main__":
    main()