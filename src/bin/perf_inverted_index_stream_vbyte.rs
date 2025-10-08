use clap::Parser;
use seismic::libbin::perf_inverted_index::*;
use seismic::match_value_fixed;
use seismic::stream_vbyte_dataset::dataset::SparseDatasetStreamVbyte;

pub fn main() {
    let args = Args::parse();
    macro_rules! run_performance_test_macro {
        ($V:ty) => {
            println!("Using {} value type", stringify!($V));
            run_performance_test_generic::<SparseDatasetStreamVbyte<$V>>(args);
        };
    }

    match_value_fixed!(args.value_type(), run_performance_test_macro);
}
