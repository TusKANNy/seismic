use std::cmp;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use half::bf16;
use half::f16;
use seismic::FixedU8Q;
use seismic::FixedU16Q;

use seismic::utils::read_from_path;
use seismic::*;

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

    /// This parameter introduces an efficiency and accuracy trade-off. The search algorithm only considers the top `query_cut` components of the query.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10)]
    query_cut: usize,

    /// This parameter introduces an efficiency and accuracy trade-off. The search algorithm skips any block which estimated dot product is greater than `heap_factor` times the smallest dot product of the top-k results in the current heap.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.7)]
    heap_factor: f32,

    /// The algorithm can score some vectors which are neighbours of the top results discovered in the previous phase. This parameter specifies how many neighbours of the top results the algorithm will score to score in the search algorithm. The knn must be computed and stored at building time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    n_knn: usize,

    /// This parameter specifies whether the list of the first component mus be first sorted by estimated dot products. This introduces an efficiency and accuracy trade-off.
    #[clap(short, long, action)]
    #[arg(default_value_t = false)]
    first_sorted: bool,

    #[clap(long, value_parser)]
    query_energy: Option<f32>,

    /// Component type: u16 (for component IDs up to 65535) or u32 (for larger component IDs)
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type: f16, bf16, f32, fixedu8, orfixedu16.
    #[clap(long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

fn run_performance_test_generic<C, D>(args: Args)
where
    C: ComponentType + serde::Serialize + for<'de> serde::Deserialize<'de>,
    D: ValueType + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let index_path = args.index_file;
    let query_cut = args.query_cut;
    let heap_factor = args.heap_factor;
    let n_runs = args.n_runs;

    let nknn = args.n_knn;

    let inverted_index: InvertedIndex<C, D> = read_from_path(index_path.unwrap().as_str()).unwrap();

    let queries = SparseDatasetMut::<C, f32>::read_bin_file(&args.query_file.unwrap()).unwrap();

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
    //eprintln!("{}", elapsed.as_micros() / (n_runs * n_queries) as u128);

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

pub fn main() {
    let args = Args::parse();

    match (args.component_type.as_str(), args.value_type.as_str()) {
        ("u16", "f16") => {
            println!("Using u16 component type with f16 value type");
            run_performance_test_generic::<u16, f16>(args);
        }
        ("u16", "bf16") => {
            println!("Using u16 component type with bf16 value type");
            run_performance_test_generic::<u16, bf16>(args);
        }
        ("u16", "f32") => {
            println!("Using u16 component type with f32 value type");
            run_performance_test_generic::<u16, f32>(args);
        }
        ("u16", "fixedu8") => {
            println!("Using u16 component type with fixedu8 value type");
            run_performance_test_generic::<u16, FixedU8Q>(args);
        }
        ("u16", "fixedu16") => {
            println!("Using u16 component type with fixedu16 value type");
            run_performance_test_generic::<u16, FixedU16Q>(args);
        }
        ("u32", "f16") => {
            println!("Using u32 component type with f16 value type");
            run_performance_test_generic::<u32, f16>(args);
        }
        ("u32", "bf16") => {
            println!("Using u32 component type with bf16 value type");
            run_performance_test_generic::<u32, bf16>(args);
        }
        ("u32", "f32") => {
            println!("Using u32 component type with f32 value type");
            run_performance_test_generic::<u32, f32>(args);
        }
        ("u32", "fixedu8") => {
            println!("Using u32 component type with fixedu8 value type");
            run_performance_test_generic::<u32, FixedU8Q>(args);
        }
        ("u32", "fixedu16") => {
            println!("Using u32 component type with fixedu16 value type");
            run_performance_test_generic::<u32, FixedU16Q>(args);
        }
        _ => {
            eprintln!(
                "Error: component-type must be either 'u16' or 'u32', value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'"
            );
            std::process::exit(1);
        }
    }
}
