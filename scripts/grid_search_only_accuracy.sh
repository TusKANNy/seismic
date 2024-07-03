index=$1
results_file=$2
queries_path=$3
gt_path=$4

exec=./target/release/perf_inverted_index



#echo -e "heap_factor\tquery_cut\ttime\trecall\tmrr@10\n" > $results_file
echo -e "heap_factor\tquery_cut\ttime\trecall\n" > $results_file


#  -i splade-cocondenser-ensembledistil.docs.kmeans_5000_350.inverted_index  -q /data6/sparse-mips-datasets/splade-cocondenser-ensembledistil.queries.bin --query-cut 10 -t 5000 -h 0.8 --n-queries 0 2> results_check_5000_350_queries_kmeans_0.8.tsv
# -t is useless

for qc in {1..10}; do
    for h in 0.6 0.7 0.8 0.9 1.0; do
        output_file="temp_${h}_${qc}"
        $exec -i "$index" -q "$queries_path" --query-cut "$qc" --heap-factor "$h" --output-path "$output_file" 2> "time_file"
        time=$(tail -n 1 "time_file")

        python scripts/accuracy.py $gt_path $output_file 2> useless_temp_2
        recall=$(tail useless_temp_2)
        echo -e "${h}\t${qc}\t${time}\t${recall}" >> $results_file
        rm $output_file
        done
    done