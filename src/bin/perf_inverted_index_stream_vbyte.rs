use clap::Parser;
use seismic::libbin::perf_inverted_index::*;
use seismic::stream_vbyte_dataset::dataset::SparseDatasetStreamVbyte;

pub fn main() {
    let args = Args::parse();

    run_performance_test_generic::<SparseDatasetStreamVbyte>(args);
}
