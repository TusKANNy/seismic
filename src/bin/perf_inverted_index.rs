use clap::Parser;
use seismic::libbin::perf_inverted_index::*;
use seismic::match_component_value;

pub fn main() {
    let args = Args::parse();

    macro_rules! run_performance_test_macro {
        ($C:ty, $V:ty) => {
            println!(
                "Using {} component type with {} value type",
                stringify!($C),
                stringify!($V)
            );
            run_performance_test_generic::<
                vectorium::PlainSparseDataset<$C, $V, vectorium::DotProduct>,
                vectorium::PlainSparseQuantizer<$C, $V, vectorium::DotProduct>,
            >(args);
        };
    }

    match_component_value!(
        args.component_type(),
        args.value_type(),
        run_performance_test_macro
    );
}
