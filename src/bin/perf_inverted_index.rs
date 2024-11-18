use std::cmp;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use half::f16;
use seismic::{InvertedIndex, SparseDataset};

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: Option<String>,

    /// The query file.
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The number of queries to evaluate.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10000)]
    n_queries: usize,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The number of runs to perform.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 1)]
    n_runs: usize,

    /// This paramenter introduces an efficiency and accuracy trade-off. The search algorithm only considers the top `query_cut` components of the query.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10)]
    query_cut: usize,

    /// This paramenter introduces an efficiency and accuracy trade-off. The search algorithm skips any block which estimated dot product is greater than `heap_factor` times the smallest dot product of the top-k results in the current heap.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.7)]
    heap_factor: f32,

    /// The algorithm can score some vectors which are neighbours of the top results discovered in the previous phase. This parameter specifies how many neighbours of the top results the algorithmw will score to score in the search algorithm. The knn must be computed and stored at building time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    n_knn: usize,

    /// This parameter specifies whether the list of the first component mus be first sorted by estimated dot products. This introduces an efficiency and accuracy trade-off.
    #[clap(short, long, action)]
    #[arg(default_value_t = false)]
    first_sorted: bool,
}

pub fn main() {
    let args = Args::parse();

    let index_path = args.index_file;
    let query_path = args.query_file;
    let query_cut = args.query_cut;
    let heap_factor = args.heap_factor;
    let n_runs = args.n_runs;

    let nknn = args.n_knn;

    let serialized: Vec<u8> = fs::read(index_path.unwrap()).unwrap();

    let inverted_index = bincode::deserialize::<InvertedIndex<f16>>(&serialized).unwrap();

    let queries = SparseDataset::<f32>::read_bin_file(&query_path.unwrap()).unwrap();

    let n_queries = cmp::min(args.n_queries, queries.len());

    println!("Searching for top-{} results", args.k);
    println!("Number of evaluated queries: {n_queries}");
    println!(
        "Avg number of non-zero components: {}",
        queries.nnz() / queries.len()
    );

    println!("Number of documents: {}", inverted_index.len());
    println!(
        "Avg number of non-zero components: {}",
        inverted_index.nnz() / inverted_index.len()
    );

    let mut results = Vec::with_capacity(n_queries);
    let time = Instant::now();
    for _ in 0..n_runs {
        results.clear();

        for (query_id, (q_components, q_values)) in queries.iter().take(n_queries).enumerate() {
            let cur_results = inverted_index.search(
                q_components,
                q_values,
                args.k,
                query_cut,
                heap_factor,
                nknn,
                args.first_sorted,
            );

            if cur_results.len() < args.k {
                println!(
                    "FAIL! The query {query_id} has only {} results.",
                    cur_results.len()
                );
            }

            results.push(cur_results);
        }
    }
    let elapsed = time.elapsed();
    println!(
        "Time {} microsecs per query",
        elapsed.as_micros() / (n_runs * n_queries) as u128
    );
    eprintln!("{}", elapsed.as_micros() / (n_runs * n_queries) as u128);

    inverted_index.print_space_usage_byte();
    // Writes results to a file in a parsable format
    let output_path = args.output_path.unwrap();
    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.iter().enumerate() {
        // Writes results to a file in a parsable format
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1,
            )
            .unwrap();
        }
    }
}
