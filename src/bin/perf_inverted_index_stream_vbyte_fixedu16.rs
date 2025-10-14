use clap::Parser;
use seismic::libbin::perf_inverted_index::*;
use seismic::match_value_fixed;
use seismic::stream_vbyte_dataset::dataset_fixedu16::SparseDatasetStreamVbyteFixedu16;

pub fn main() {
    let args = Args::parse();
    run_performance_test_generic::<SparseDatasetStreamVbyteFixedu16>(args);
}
