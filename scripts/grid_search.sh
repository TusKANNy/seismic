# index=$1
# results_file=$2
# queries_path=$3
# gt_path=$4
# qrels_path=$5
# original_queries_path=$6

# exec=./target/release/perf_inverted_index

# echo -e "heap_factor\tquery_cut\ttime\trecall\tmrr@10\n" > $results_file

# for qc in 1 2 3 4 5 6 7 8 9 10 
#     do
#     for h in 0.6 0.7 0.8 0.9 1.0
#         do
#         output_file="temp_${h}_${qc}_2"
#         echo $output_file
#         rm $output_file
#         #$exec -i $index -q $queries_path --query-cut $qc --heap-factor $h 2> $output_file
#         $exec -i $index -q $queries_path --query-cut $qc --heap-factor $h --output-path $output_file 2> "time_file"
#         time=$(tail -n 1 "time_file")
#         python scripts/recall_and_mrr.py $gt_path $output_file $qrels_path $original_queries_path 2> useless_temp_2
#         recall=$(tail useless_temp_2)
#         echo -e "${h}\t${qc}\t${time}\t${recall}" >> $results_file

#         done
#     done

#!/bin/bash

index=$1
results_file=$2
queries_path=$3
gt_path=$4
qrels_path=$5
original_queries_path=$6
exec=./target/release/perf_inverted_index

echo -e "heap_factor\tquery_cut\ttime\trecall\tmrr@10\n" > "$results_file"

for qc in {1..15}; do
    for h in 0.6 0.7 0.8 0.9 1.0; do
        output_file="temp_${h}_${qc}_2"
        echo "$output_file"
        rm -f "$output_file"  # Ensure the file is removed if it exists
        
        # $exec -i "$index" -q "$queries_path" --query-cut "$qc" --heap-factor "$h" 2> "$output_file"
        $exec -i "$index" -q "$queries_path" --query-cut "$qc" --heap-factor "$h" --output-path "$output_file" 2> "time_file"
        time=$(tail -n 1 "time_file")
        
        python scripts/compute_accuracy_and_mrr.py --gt-path "$gt_path" --run-path "$output_file" --qrels-path "$qrels_path" --original-queries-path "$original_queries_path" 2> useless_temp_2
        recall=$(tail -n 1 useless_temp_2)
        
        echo -e "${h}\t${qc}\t${time}\t${recall}" >> "$results_file"
        rm -f "$output_file"  # Ensure the file is removed if it exists

    done
done