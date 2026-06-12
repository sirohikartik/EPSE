#!/usr/bin/env bash
# run_experiment.sh
# -----------------
# Runs EPSE clustering for all thresholds listed in Issue #9,
# then calls analyze_clusters.py to compare results.
#
# Usage:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh
#
# Optional: pass a custom FASTA file path as first argument:
#   ./run_experiment.sh /path/to/my.fasta

set -e  # stop on any error

FASTA="${1:-uniprot_sprot.fasta}"
BINARY="./build/epse"
THRESHOLDS=(0.90 0.92 0.94 0.96 0.98 0.99)

# --- Check the binary exists ---
if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Build first with:"
    echo "  mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# --- Check the FASTA file exists ---
if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA file '$FASTA' not found."
    echo "Download SwissProt: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    echo "Then: gunzip uniprot_sprot.fasta.gz"
    exit 1
fi

echo "========================================"
echo "  EPSE Threshold Experiment"
echo "  FASTA : $FASTA"
echo "  Runs  : ${THRESHOLDS[*]}"
echo "========================================"
echo ""

for T in "${THRESHOLDS[@]}"; do
    echo "-------- Running threshold $T --------"
    "$BINARY" "$T" "$FASTA"
    echo ""
done

echo "========================================"
echo "  All runs complete! Analyzing results..."
echo "========================================"

python3 analyze_clusters.py --all