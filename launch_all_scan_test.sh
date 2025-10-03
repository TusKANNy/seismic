data_path=/data2/knn_datasets/sparse_datasets/msmarco_v1_passage/cocondenser/data
document_path=$data_path/documents.bin
query_path=$data_path/queries.bin

# $compression_results_dir=codec_results
# $log_results_dir=codec_logs

# bash test_compression_codecs.sh $document_path $query_path $compression_results_dir $log_results_dir


$baseline_streamvbyte_log_dir=baseline_streamvbyte_logs
$baseline_streamvbyte_results_dir=baseline_streamvbyte_results

bash test_baseline_streamvbyte_dataset.sh $document_path $query_path $baseline_streamvbyte_results_dir  $baseline_streamvbyte_log_dir


# $partition_results_dir=partition_results
# $partition_log_dir=partition_logs

# bash test_partitioned_dataset.sh $document_path $query_path $partition_results_dir $partition_log_dir

