#!/usr/bin/env python3
"""
analyze_clusters.py
--------------------
Reads centroids.csv from one or more threshold runs and prints cluster statistics.

Usage:
    # Analyze a single threshold run:
    python analyze_clusters.py 0.94

    # Analyze and compare ALL threshold runs at once:
    python analyze_clusters.py --all

    # Specify a custom data directory:
    python analyze_clusters.py --dir .data/threshold_0.94
"""

import os
import sys
import csv
import argparse
import statistics


def analyze(centroids_csv_path: str, threshold_label: str = "") -> dict:
    """Read a centroids.csv and return cluster stats."""
    counts = []

    with open(centroids_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts.append(int(row["count"]))

    if not counts:
        print(f"  [WARNING] No clusters found in {centroids_csv_path}")
        return {}

    num_clusters    = len(counts)
    largest         = max(counts)
    smallest        = min(counts)
    total_proteins  = sum(counts)
    avg             = total_proteins / num_clusters
    med             = statistics.median(counts)
    singleton_count = sum(1 for c in counts if c == 1)
    giant_threshold = total_proteins * 0.10  # cluster with >10% of all proteins
    giant_count     = sum(1 for c in counts if c > giant_threshold)

    # Histogram buckets
    buckets = {"1": 0, "2-10": 0, "11-100": 0, "101-1000": 0, "1001+": 0}
    for c in counts:
        if c == 1:          buckets["1"]       += 1
        elif c <= 10:       buckets["2-10"]    += 1
        elif c <= 100:      buckets["11-100"]  += 1
        elif c <= 1000:     buckets["101-1000"]+= 1
        else:               buckets["1001+"]   += 1

    label = f"Threshold {threshold_label}" if threshold_label else centroids_csv_path

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Total proteins   : {total_proteins:,}")
    print(f"  Num clusters     : {num_clusters:,}")
    print(f"  Largest cluster  : {largest:,}")
    print(f"  Smallest cluster : {smallest:,}")
    print(f"  Average size     : {avg:.1f}")
    print(f"  Median size      : {med:.1f}")
    print(f"  Singleton (1)    : {singleton_count:,}  ({100*singleton_count/num_clusters:.1f}%)")
    print(f"  Giant (>10%)     : {giant_count}")
    print(f"\n  Size histogram:")
    for bucket, cnt in buckets.items():
        bar = "█" * min(40, cnt // max(1, num_clusters // 40))
        print(f"    {bucket:>10}  : {cnt:>6}  {bar}")

    return {
        "threshold":        threshold_label,
        "num_clusters":     num_clusters,
        "total_proteins":   total_proteins,
        "largest":          largest,
        "average":          round(avg, 1),
        "median":           med,
        "singletons":       singleton_count,
        "giants":           giant_count,
    }


def compare(results: list):
    """Print a comparison table of all threshold results."""
    if len(results) < 2:
        return

    print(f"\n\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"  {'Threshold':>10}  {'Clusters':>10}  {'Largest':>10}  {'Avg':>8}  {'Median':>8}  {'Singles':>8}"
    print(header)
    print(f"  {'-'*64}")
    for r in results:
        print(f"  {r['threshold']:>10}  {r['num_clusters']:>10,}  {r['largest']:>10,}  "
              f"{r['average']:>8}  {r['median']:>8}  {r['singletons']:>8,}")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    # Best = fewest giants AND not too many singletons (fewest singletons as tiebreak)
    scored = sorted(results, key=lambda x: (x["giants"], x["singletons"]))
    best = scored[0]
    print(f"  → Threshold {best['threshold']} looks best:")
    print(f"    {best['giants']} giant cluster(s), {best['singletons']:,} singletons, "
          f"{best['num_clusters']:,} total clusters.")


def main():
    parser = argparse.ArgumentParser(description="Analyze EPSE cluster results")
    parser.add_argument("threshold", nargs="?", help="Single threshold to analyze (e.g. 0.94)")
    parser.add_argument("--all",  action="store_true", help="Analyze all threshold runs in .data/")
    parser.add_argument("--dir",  help="Path to a specific threshold data directory")
    args = parser.parse_args()

    results = []

    if args.dir:
        path = os.path.join(args.dir, "centroids.csv")
        label = args.dir
        r = analyze(path, label)
        if r:
            results.append(r)

    elif args.all:
        data_root = ".data"
        if not os.path.isdir(data_root):
            print("ERROR: .data/ directory not found. Run ./epse first.")
            sys.exit(1)
        dirs = sorted(d for d in os.listdir(data_root) if d.startswith("threshold_"))
        if not dirs:
            print("No threshold runs found in .data/")
            sys.exit(1)
        for d in dirs:
            path = os.path.join(data_root, d, "centroids.csv")
            if os.path.isfile(path):
                label = d.replace("threshold_", "")
                r = analyze(path, label)
                if r:
                    results.append(r)

    elif args.threshold:
        path = os.path.join(".data", f"threshold_{args.threshold}", "centroids.csv")
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found. Did you run ./epse {args.threshold}?")
            sys.exit(1)
        r = analyze(path, args.threshold)
        if r:
            results.append(r)

    else:
        parser.print_help()
        sys.exit(0)

    compare(results)
    print()


if __name__ == "__main__":
    main()