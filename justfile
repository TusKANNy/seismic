PERF := "taskset -c 0-7 perf stat -e cycles,branches,branch-misses,LLC-loads,LLC-load-misses"

compile:
    export GAIN=1;
    export SEISMIC_N_PARTITIONS=128;
    export SEISMIC_N_COMPONENT_BITS=8;
    RUSTFLAGS='-C target-cpu=native' cargo build --release

build_f32:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/build_inverted_index --input-file /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/documents.bin --output-file /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8 --n-postings 6000 --summary-energy 0.5 --centroid-fraction 0.1 --knn 0 --clustering-algorithm random-kmeans-inverted-index-approx --kmeans-doc-cut 15 --min-cluster-size 10 --max-fraction 3.0 --value-type f32 --pruning-strategy global-threshold

perf_f32:
    ./target/release/perf_inverted_index -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8.index.seismic --heap-factor 0.9 --query-cut 8 --first-sorted  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type f32  -o res_f32.tsv

build_u16:
    ./target/release/build_inverted_index --input-file /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/documents.bin --output-file /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8f --n-postings 6000 --summary-energy 0.5 --centroid-fraction 0.1 --knn 0 --clustering-algorithm random-kmeans-inverted-index-approx --kmeans-doc-cut 15 --min-cluster-size 10 --max-fraction 3.0 --value-type fixedu8 --pruning-strategy global-threshold

perf_u16:
    {{PERF}} ./target/release/perf_inverted_index -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8f.index.seismic --heap-factor 0.9 --query-cut 8 --first-sorted  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8  -o res_f8.tsv

build_partitioned: 
    ./target/release/convert_inverted_index_partitioned --index-file /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8.index.seismic -o /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8 --value-type fixedu8

perf_partitioned:
    ./target/release/perf_inverted_index_stream_vbyte -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8.128_part_8_compbits.index.seismic --heap-factor 0.9 --query-cut 8 --first-sorted  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8  --component-type u16 -o res_partitioned.tsv

build_stream: 
    ./target/release/convert_inverted_index_stream_vbyte --index-file /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8.index.seismic -o /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8 --value-type fixedu8

perf_stream:
    {{PERF}} ./target/release/perf_inverted_index_stream_vbyte -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8_streamvbyte.index.seismic --heap-factor 0.9 --query-cut 8 --first-sorted  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8  -o res_vbyte.tsv

flame_stream:
    flamegraph --root --title "Inverted Index Stream VByte" -- ./target/release/perf_inverted_index_stream_vbyte -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8_streamvbyte.index.seismic --heap-factor 0.9 --query-cut 8 --first-sorted  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8  -o res_vbyte.tsv --n-runs 20

build_stream_merge: 
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/convert_inverted_index_stream_vbyte_merge --index-file /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8.index.seismic -o /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8 --value-type fixedu8  

perf_stream_merge:
    {{PERF}} ./target/release/perf_inverted_index_stream_vbyte_merge -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8_streamvbyte_merge.index.seismic --heap-factor 0.9 --query-cut 8 --first-sorted  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8 -o res_vbyte_merge.tsv
    diff -y --suppress-common-lines res_vbyte.tsv res_vbyte_merge.tsv | tee diff.txt | wc -l

bench_dot_stream:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/bench_dot_stream_vbyte -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8_streamvbyte.index.seismic -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8 --n-docs 1000

bench_dot_u16:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/bench_dot_u16 -i /data2/rossano/inference_less_big/indexes/global_threshold_index_c-f_0.1_c-a_r-k-i-a_k-d-c_15_knn_0_m-f_3.0_m-c-s_10_n-p_6000_p-s_g-t_s-e_0.5_v-t_fixedu8f.index.seismic  -q /data2/knn_datasets/sparse_datasets/msmarco_v1_passage/inference_less/data/queries.bin --value-type fixedu8 --n-docs 1000

compare:
    diff -y --suppress-common-lines res_vbyte.tsv res_vbyte_merge.tsv | tee diff.txt | wc -l