use clap::Parser;
use seismic::libbin::convert_inverted_index::*;
use seismic::match_value;
use seismic::partitioned_dataset::dataset::SparseDatasetPartitioned;

pub fn main() {
    const N_PARTITIONS: usize = envparse::parse_env!("SEISMIC_N_PARTITIONS" as usize);
    const N_COMPONENT_BITS: usize = envparse::parse_env!("SEISMIC_N_COMPONENT_BITS" as usize);
    let mut args = Args::parse();

    args.output_file = args
        .output_file
        .map(|s| format!("{}.{}_part_{}_compbits", s, N_PARTITIONS, N_COMPONENT_BITS));

    macro_rules! convert_index_macro {
        ($V:ty) => {
            println!("Using {} value type", stringify!($V));
            convert_index_from_f32::<SparseDatasetPartitioned<N_PARTITIONS, N_COMPONENT_BITS, $V>>(
                args,
            );
        };
    }

    match_value!(args.value_type(), convert_index_macro);
}
