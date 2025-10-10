export GAIN=approx_1
export SEISMIC_N_PARTITIONS=128 && export SEISMIC_N_COMPONENT_BITS=8
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/baseline_streamvbyte_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/baseline_streamvbyte_f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/partitioned_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/partitioned_f16.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/streamvbyte_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/compression_f8.toml
python scripts/run_grid_search.py --exp experiments/ecir2026/spladev3/compression_f16.toml