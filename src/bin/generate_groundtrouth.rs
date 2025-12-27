use clap::Parser;
use std::fs::File;
use std::io::Write;

use indicatif::ParallelProgressIterator;
use rayon::iter::ParallelIterator;

use vectorium::{Dataset, Distance, read_seismic_format};

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

    let dataset =
        read_seismic_format::<u32, f32, vectorium::DotProduct>(&args.input_file.unwrap()).unwrap();
    let queries =
        read_seismic_format::<u32, f32, vectorium::DotProduct>(&args.query_file.unwrap()).unwrap();
    let k = args.k;
    let output_path = args.output_path.unwrap();

    let results: Vec<_> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .map(|query| dataset.search(query, k))
        .collect();

    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.iter().enumerate() {
        // Writes results to a file in a parsable format
        for (idx, item) in result.iter().enumerate() {
            let score = item.distance.distance();
            let doc_id = item.vector;
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .unwrap();
        }
    }
}
