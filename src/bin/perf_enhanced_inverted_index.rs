use std::cmp;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use half::f16;
use seismic::SearchResult;
use seismic::SeismicIndex;
use seismic::json_utils::read_queries;
use vectorium::{DotProduct, IndexSerializer, ScalarSparseQuantizer, SpaceUsage, SparseDataset};

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the serialized index file (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, value_parser)]
    index_file: Option<String>,

    /// Query file in `.jsonl` format (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// Output file with the ranked results (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// Number of queries to evaluate; see docs/RustUsage.md#using-the-rust-code for recommended scaling.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10000)]
    n_queries: usize,

    /// Number of top-k results to retrieve; tuning hints appear in docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// Number of runs to average; see docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 1)]
    n_runs: usize,

    /// This parameter introduces an efficiency/accuracy trade-off: only the top `query_cut` components are considered (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10)]
    query_cut: usize,

    /// Controls block skipping based on estimated dot products (see docs/RustUsage.md#using-the-rust-code for the accuracy impact).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.7)]
    heap_factor: f32,

    /// Number of neighbors of top results to rescore; see docs/RustUsage.md#using-the-rust-code for the intended accuracy trade-off.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    n_knn: usize,

    /// Whether to sort the first component by estimated dot products before search (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, action)]
    #[arg(default_value_t = false)]
    first_sorted: bool,

    /// Value type: f16 or f32; see docs/RustUsage.md#using-the-rust-code for quantization choices.
    #[clap(short, long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

pub fn main() {
    let args = Args::parse();

    let index_path = args.index_file;
    let query_path = args.query_file;
    let query_cut = args.query_cut;
    let heap_factor = args.heap_factor;
    let n_runs = args.n_runs;

    let nknn = args.n_knn;

    type Encoder = ScalarSparseQuantizer<u16, f32, f16, DotProduct>;
    type Dataset = SparseDataset<Encoder>;

    let inverted_index: SeismicIndex<Dataset> =
        SeismicIndex::load_index(index_path.unwrap().as_str())
            .unwrap_or_else(|err| panic!("Failed to load index: {err:?}"));

    let queries = read_queries(&query_path.unwrap());

    let n_queries = cmp::min(args.n_queries, queries.len());

    println!("Searching for top-{} results", args.k);
    println!("Number of evaluated queries: {n_queries}");

    println!("Number of documents: {}", inverted_index.len());
    println!(
        "Avg number of non-zero components: {}",
        inverted_index.nnz() / inverted_index.len()
    );

    let mut results: Vec<Vec<SearchResult>> = Vec::with_capacity(n_queries);
    let time = Instant::now();
    for _ in 0..n_runs {
        results.clear();
        for (query_id, q_components, q_values) in queries.iter().take(n_queries) {
            let cur_results = inverted_index.search(
                query_id,
                q_components,
                q_values,
                args.k,
                query_cut,
                heap_factor,
                nknn,
                args.first_sorted,
                false,
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

    let space_usage = inverted_index.space_usage_bytes();
    eprintln!(
        "{}\t{}",
        elapsed.as_micros() / (n_runs * n_queries) as u128,
        space_usage
    );

    inverted_index.print_space_usage_byte();

    // Writes results to a file in a parsable format
    let output_path = args.output_path.unwrap();
    let mut output_file = File::create(output_path).unwrap();

    for current_result in results.iter() {
        for (idx, result) in current_result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{}\t{}\t{}\t{}",
                result.query_id,
                result.doc_id,
                idx + 1,
                result.score,
            )
            .unwrap();
        }
    }
}
