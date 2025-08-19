use seismic::partitioned_dataset::dataset::SparseDatasetPartitioned;
use seismic::utils::{read_from_path, write_to_path};
use seismic::*;

use clap::Parser;
use fixed::FixedU16;
use fixed::types::extra::U14;
use std::time::Instant;

// TODO:
// - add control to the Rayon's number of threads

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the input file
    #[clap(short, long, value_parser)]
    index_file: Option<String>,

    /// The path of the output file. The extension will encode the values of the building parameters.
    #[clap(short, long, value_parser)]
    output_file: Option<String>,
}

pub fn main() {
    const N_PARTITIONS: usize = envparse::parse_env!("SEISMIC_N_PARTITIONS" as usize);
    const N_COMPONENT_BITS: usize = envparse::parse_env!("SEISMIC_N_COMPONENT_BITS" as usize);
    let args = Args::parse();
    let index_path = args.index_file;

    println!("Loading inverted index...");

    let inverted_index: InvertedIndex<SparseDataset<u16, f32>> =
        read_from_path(index_path.unwrap().as_str()).unwrap();

    println!("Number of Vectors: {}", inverted_index.len());
    println!("Number of Dimensions: {}", inverted_index.dim());

    println!(
        "Avg number of components: {:.2}",
        inverted_index.nnz() as f32 / inverted_index.len() as f32
    );

    let time = Instant::now();

    println!("Converting the inverted index...");
    let inverted_index_partitioned = InvertedIndex::<
        SparseDatasetPartitioned<N_PARTITIONS, N_COMPONENT_BITS, FixedU16<U14>>,
    >::from_inverted_index(inverted_index);

    let elapsed = time.elapsed();
    println!(
        "Time to convert {} secs (before serializing)",
        elapsed.as_secs()
    );

    let path = format!(
        "{}.{}_part_{}_compbits_index.seismic",
        args.output_file.unwrap(),
        N_PARTITIONS,
        N_COMPONENT_BITS
    );
    println!("Saving ... {}", path);
    let r = write_to_path(inverted_index_partitioned, path.as_str());

    println!("{:?}", r);

    let elapsed = time.elapsed();
    println!("Time to build {} secs", elapsed.as_secs());
}
