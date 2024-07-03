use clap::Parser;
use std::fs::File;
use std::io::Write;

use indicatif::ParallelProgressIterator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use seismic::SparseDataset;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The binary file with dataset vectors
    #[clap(short, long, value_parser)]
    input_file: Option<String>,

    /// The binary file with query vectors
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// The number of results to report for each query
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,
}

pub fn main() {
    let args = Args::parse();

    let dataset = SparseDataset::<f32>::read_bin_file(&args.input_file.unwrap()).unwrap();
    let queries = SparseDataset::<f32>::read_bin_file(&args.query_file.unwrap()).unwrap();
    let k = args.k;
    let output_path = args.output_path.unwrap();

    let results: Vec<_> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .map(|(q_components, q_values)| dataset.search(q_components, q_values, k))
        .collect();

    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.iter().enumerate() {
        // Writes results to a file in a parsable format
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .unwrap();
        }
    }
}
