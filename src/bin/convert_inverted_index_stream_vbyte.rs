use clap::Parser;
use seismic::libbin::convert_inverted_index::*;
use seismic::stream_vbyte_dataset::dataset::SparseDatasetStreamVbyte;

pub fn main() {
    let args = Args::parse();

    convert_index_from_f32::<SparseDatasetStreamVbyte>(args);
}
