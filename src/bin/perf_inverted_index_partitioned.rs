use clap::Parser;
use seismic::libbin::perf_inverted_index::*;
use seismic::match_value;
use seismic::partitioned_dataset::dataset::SparseDatasetPartitioned;

pub fn main() {
    const N_PARTITIONS: usize = envparse::parse_env!("SEISMIC_N_PARTITIONS" as usize);
    const N_COMPONENT_BITS: usize = envparse::parse_env!("SEISMIC_N_COMPONENT_BITS" as usize);

    let args = Args::parse();

    macro_rules! run_performance_test_macro {
        ($V:ty) => {
            println!("Using {} value type", stringify!($V));
            run_performance_test_generic::<
                SparseDatasetPartitioned<N_PARTITIONS, N_COMPONENT_BITS, $V>,
            >(args);
        };
    }

    match_value!(args.value_type(), run_performance_test_macro);
}
