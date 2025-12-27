use clap::Parser;
use seismic::libbin::perf_inverted_index::*;
use seismic::match_component_value;
use vectorium::{DotVByteFixedU8Quantizer, PackedDataset};

pub fn main() {
    let args = Args::parse();

    if args.value_type() == "dotvbyte" {
        if args.component_type() != "u16" {
            eprintln!("Error: value-type 'dotvbyte' is only supported with component-type 'u16'");
            std::process::exit(1);
        }

        println!("Using u16 component type with dotvbyte value type");
        run_performance_test_generic::<PackedDataset<DotVByteFixedU8Quantizer>, DotVByteFixedU8Quantizer>(
            args,
        );
        return;
    }

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
