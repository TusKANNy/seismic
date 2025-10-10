export GAIN=approx_1
export SEISMIC_N_PARTITIONS=128 && export SEISMIC_N_COMPONENT_BITS=8
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/baseline_streamvbyte_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/baseline_streamvbyte_f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/partitioned_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/partitioned_f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/streamvbyte_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/compression_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/inference_less_big/compression_f16.toml