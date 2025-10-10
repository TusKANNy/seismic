use clap::Parser;
use seismic::libbin::convert_inverted_index::*;

use seismic::stream_vbyte_dataset::dataset_fixedu8::SparseDatasetStreamVbyteFixedu8;

pub fn main() {
    let mut args = Args::parse();
    args.output_file = args.output_file.map(|s| format!("{}_streamvbyte", s));
    convert_index_from_f32::<SparseDatasetStreamVbyteFixedu8>(args);
}
