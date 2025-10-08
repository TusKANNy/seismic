use clap::Parser;
use seismic::libbin::convert_inverted_index::*;
use seismic::match_value_fixed;
use seismic::stream_vbyte_dataset::dataset::SparseDatasetStreamVbyte;

pub fn main() {
    let mut args = Args::parse();
    args.output_file = args.output_file.map(|s| format!("{}_streamvbyte", s));

    macro_rules! convert_index_macro {
        ($V:ty) => {
            println!("Using {} value type", stringify!($V));
            convert_index_from_f32::<SparseDatasetStreamVbyte<$V>>(args);
        };
    }

    match_value_fixed!(args.value_type(), convert_index_macro);
}
