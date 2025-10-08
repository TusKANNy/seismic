export GAIN=approx_1
export SEISMIC_N_PARTITIONS=128 && export SEISMIC_N_COMPONENT_BITS=8
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/baseline_streamvbyte_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/baseline_streamvbyte_f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/partitioned_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/partitioned_f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/streamvbyte.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/compression_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/cocondenser/compression_f16.toml