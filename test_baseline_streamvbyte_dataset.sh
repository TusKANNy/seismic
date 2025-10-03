#!/bin/bash

# Script to test permutation strategies available in search_baseline_streamvbyte_dataset
# Usage: ./test_baseline_streamvbyte_dataset.sh <input_file> <query_file> [output_dir] [log_file]

set -e  # Exit on any error

# Default configuration
DEFAULT_OUTPUT_DIR="./baseline_streamvbyte_results"
DEFAULT_LOG_DIR="./baseline_streamvbyte_log"
DEFAULT_K=10
DEFAULT_N_QUERIES=100

# Input arguments
INPUT_FILE="$1"
QUERY_FILE="$2"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"
LOG_DIR="${4:-$DEFAULT_LOG_DIR}"

# Validate input arguments
if [[ -z "$INPUT_FILE" || -z "$QUERY_FILE" ]]; then
    echo "Usage: $0 <input_file> <query_file> [output_dir] [log_dir]"
    echo "Example: $0 documents.bin queries.bin ./results ./baseline_streamvbyte_log"
    exit 1
fi

# Check if input files exist
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

if [[ ! -f "$QUERY_FILE" ]]; then
    echo "Error: Query file '$QUERY_FILE' not found!"
    exit 1
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Binary path
BINARY="./target/release/search_baseline_streamvbyte_dataset"

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary '$BINARY' not found! Please build with: cargo build --release --bin search_baseline_streamvbyte_dataset"
    exit 1
fi

# Permutation strategies to test
PERMUTATION_MODES=("none" "metis" "graph-bisection")

echo "========================================="
echo "Starting baseline StreamVByte dataset benchmark"
echo "========================================="
echo "Input file: $INPUT_FILE"
echo "Query file: $QUERY_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Permutation modes: ${PERMUTATION_MODES[*]}"
echo "K results: $DEFAULT_K"
echo "Number of queries: $DEFAULT_N_QUERIES"
echo "========================================="
echo

# Main testing loop
for permutation in "${PERMUTATION_MODES[@]}"; do
    echo "Testing permutation strategy: $permutation"
    
    output_file="$OUTPUT_DIR/results_${permutation}.tsv"
    log_file="$LOG_DIR/${permutation}.tsv"
    
    # Remove existing log file if it exists
    if [[ -f "$log_file" ]]; then
        echo "Removing existing log file for $permutation: $log_file"
        rm "$log_file"
    fi
    
    # Build command
    cmd="$BINARY"
    cmd+=" --input-file $INPUT_FILE"
    cmd+=" --query-file $QUERY_FILE"
    cmd+=" -k $DEFAULT_K"
    cmd+=" -o $output_file"
    cmd+=" -l $log_file"
    cmd+=" --n-queries $DEFAULT_N_QUERIES"
    cmd+=" --permutation $permutation"
    
    echo "Running: $cmd"
    
    # Execute command
    if eval "$cmd"; then
        echo "✅ Successfully tested permutation strategy: $permutation"
        echo "   Results saved to: $output_file"
        echo "   Log saved to: $log_file"
    else
        echo "❌ Failed to test permutation strategy: $permutation"
        # Continue with next permutation instead of exiting
        continue
    fi
    
    echo "---"
done

echo
echo "========================================="
echo "Benchmark completed!"
echo "========================================="
echo "Results summary:"
echo "- Output files: $OUTPUT_DIR/results_*.tsv"
echo "- Log files: $LOG_DIR/*.tsv"
echo "- Tested permutation strategies: ${#PERMUTATION_MODES[@]} total experiments"
echo