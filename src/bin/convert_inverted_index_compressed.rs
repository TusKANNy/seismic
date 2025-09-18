use clap::Parser;
use seismic::compressed_dataset::SparseDatasetCompressed;
use seismic::libbin::convert_inverted_index::*;
use seismic::match_component_value;

pub fn main() {
    let args = Args::parse();

    macro_rules! convert_index_macro {
        ($C:ty, $V:ty) => {
            println!(
                "Using {} component type with {} value type",
                stringify!($C),
                stringify!($V)
            );
            convert_index_from_f32::<SparseDatasetCompressed<$C, $V>>(args);
        };
    }

    match_component_value!(
        args.component_type(),
        args.value_type(),
        convert_index_macro
    );
}
