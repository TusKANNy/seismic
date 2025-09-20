#!/bin/bash

# Script to test all compression codecs available in search_compressed_dataset
# Usage: ./test_compression_codecs.sh <input_file> <query_file> [output_dir] [log_file]

set -e  # Exit on any error

# Default parameters
DEFAULT_OUTPUT_DIR="./test_results"
DEFAULT_LOG_DIR="./codec_log"
DEFAULT_K=10
DEFAULT_N_QUERIES=100

# Parse arguments
INPUT_FILE="$1"
QUERY_FILE="$2"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"
LOG_DIR="${4:-$DEFAULT_LOG_DIR}"

# Validate required arguments
if [[ -z "$INPUT_FILE" || -z "$QUERY_FILE" ]]; then
    echo "Usage: $0 <input_file> <query_file> [output_dir] [log_dir]"
    echo "Example: $0 documents.bin queries.bin ./results ./codec_log"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Path to the binary
BINARY="./target/release/search_compressed_dataset"

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "Binary not found at $BINARY. Building..."
    cargo build --release --bin search_compressed_dataset
fi

# Array of codecs to test
#CODECS=("gamma" "delta" "v-byte-le" "v-byte-be" "zeta")


CODECS=("v-byte-le" "v-byte-be")
echo "========================================="
echo "Starting compression codec benchmark"
echo "========================================="
echo "Input file: $INPUT_FILE"
echo "Query file: $QUERY_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Codecs to test: ${CODECS[*]}"
echo "K results: $DEFAULT_K"
echo "Number of queries: $DEFAULT_N_QUERIES"
echo "========================================="
echo

# Test each codec
for codec in "${CODECS[@]}"; do
    echo "Testing codec: $codec"
    
    output_file="$OUTPUT_DIR/results_${codec}.tsv"
    log_file="$LOG_DIR/${codec}.tsv"
    
    # Remove existing log file for this codec to start fresh
    if [[ -f "$log_file" ]]; then
        echo "Removing existing log file for $codec: $log_file"
        rm "$log_file"
    fi
    
    # Build command based on codec
    cmd="$BINARY"
    cmd+=" --input-file $INPUT_FILE"
    cmd+=" --query-file $QUERY_FILE"
    cmd+=" -k $DEFAULT_K"
    cmd+=" -o $output_file"
    cmd+=" -l $log_file"
    cmd+=" -c $codec"
    cmd+=" --n-queries $DEFAULT_N_QUERIES"
    
    # For zeta, we don't specify --zeta-k to use auto mode (None)
    # This is already the default behavior
    
    echo "Running: $cmd"
    
    # Execute the command
    if eval "$cmd"; then
        echo "✅ Successfully tested $codec codec"
        echo "   Results saved to: $output_file"
        echo "   Log saved to: $log_file"
    else
        echo "❌ Failed to test $codec codec"
        # Continue with next codec instead of exiting        ./test_compression_codecs.sh /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/cocondenser/data/documents.bin /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin ./my_results ./my_codec_logs
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
echo "- Benchmark log: $LOG_FILE"
echo
