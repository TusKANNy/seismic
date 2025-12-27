use clap::Parser;
use seismic::libbin::build_inverted_index::*;
use seismic::match_component_value;

pub fn main() {
    let args = Args::parse();

    macro_rules! build_index_macro {
        ($C:ty, $V:ty) => {
            println!(
                "Using {} component type with {} value type",
                stringify!($C),
                stringify!($V)
            );
            build_index_generic::<$C, $V>(args);
        };
    }

    match_component_value!(args.component_type(), args.value_type(), build_index_macro);
}
