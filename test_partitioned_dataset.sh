#!/bin/bash

# Script to test permutation strategies available in search_partitioned_dataset
# Usage: ./test_partitioned_dataset.sh <input_file> <query_file> [output_dir] [log_file]

set -e  # Exit on any error


DEFAULT_OUTPUT_DIR="./test_results_partitioned"
DEFAULT_LOG_DIR="./log_partitioned"
DEFAULT_K=10
DEFAULT_N_QUERIES=100


INPUT_FILE="$1"
QUERY_FILE="$2"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"
LOG_DIR="${4:-$DEFAULT_LOG_DIR}"


if [[ -z "$INPUT_FILE" || -z "$QUERY_FILE" ]]; then
    echo "Usage: $0 <input_file> <query_file> [output_dir] [log_dir]"
    echo "Example: $0 documents.bin queries.bin ./results ./log_partitioned"
    exit 1
fi


if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

if [[ ! -f "$QUERY_FILE" ]]; then
    echo "Error: Query file '$QUERY_FILE' not found!"
    exit 1
fi


mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"


BINARY="./target/release/search_partitioned_dataset"


if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary '$BINARY' not found! Please build with: cargo build --release --bin search_partitioned_dataset"
    exit 1
fi


PERMUTATION_MODES=("graph-bisection" "none")

echo "========================================="
echo "Starting partitioned dataset benchmark"
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


for permutation in "${PERMUTATION_MODES[@]}"; do
    echo "Testing permutation strategy: $permutation"
    
    output_file="$OUTPUT_DIR/results_${permutation}.tsv"
    log_file="$LOG_DIR/${permutation}.tsv"
    

    if [[ -f "$log_file" ]]; then
        echo "Removing existing log file for $permutation: $log_file"
        rm "$log_file"
    fi
    

    cmd="$BINARY"
    cmd+=" --input-file $INPUT_FILE"
    cmd+=" --query-file $QUERY_FILE"
    cmd+=" -k $DEFAULT_K"
    cmd+=" -o $output_file"
    cmd+=" -l $log_file"
    cmd+=" --n-queries $DEFAULT_N_QUERIES"
    cmd+=" --permutation $permutation"
    
    echo "Running: $cmd"
    

    if eval "$cmd"; then
        echo "✅ Successfully tested permutation strategy: $permutation"
        echo "   Results saved to: $output_file"
        echo "   Log saved to: $log_file"
    else
        echo "❌ Failed to test permutation strategy: $permutation"

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