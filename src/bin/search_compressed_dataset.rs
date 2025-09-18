use clap::Parser;
use seismic::{
    FixedU16Q, FromDatasetGenericF32, SpaceUsage, SparseDatasetTrait,
    compressed_dataset::SparseDatasetCompressed, sparse_dataset::SparseDatasetMut,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the input file
    #[clap(short, long, value_parser)]
    input_file: Option<String>,

    /// The path of the query file
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// The number of results to report for each query
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The output file to write the results
    #[clap(short, long, value_parser)]
    output_path: Option<String>,
}

pub fn main() {
    let args = Args::parse();

    let input_file = match args.input_file {
        Some(file) => file,
        None => {
            eprintln!("Error: Input file is required. Use -i <path>");
            std::process::exit(1);
        }
    };

    println!("Reading the queries...");
    let queries = SparseDatasetMut::<u16, f32>::read_bin_file(&args.query_file.unwrap()).unwrap();

    println!("Loading generic dataset from: {}", input_file);
    let dataset_generic = match SparseDatasetMut::<u16, f32>::read_bin_file(&input_file) {
        Ok(dataset) => dataset,
        Err(e) => {
            eprintln!("Error reading dataset: {}", e);
            std::process::exit(1);
        }
    };

    let k = args.k;

    let start = Instant::now();
    let results: Vec<_> = queries
        .dataset_iter()
        .take(5)
        .map(|(query_components, query_values)| {
            dataset_generic.search(
                query_components
                    .iter()
                    .zip(query_values)
                    .map(|(&c, &v)| (c, v)),
                k,
            )
        })
        .collect();

    let duration = start.elapsed();

    println!("Total time taken with FP dataset: {:?}", duration);
    println!(
        "Time per query with FP dataset: {:?}",
        duration / results.len() as u32
    );
    println!(
        "Average time per document with FP dataset: {:?}",
        duration / (results.len() as u32 * dataset_generic.len() as u32)
    );

    println!("=== Generic Dataset Info ===");
    println!("Number of Vectors: {}", dataset_generic.len());
    println!("Number of Dimensions: {}", dataset_generic.dim());
    println!(
        "Avg number of components: {:.2}",
        dataset_generic.nnz() as f32 / dataset_generic.len() as f32
    );
    println!("Total non-zero components: {}", dataset_generic.nnz());
    println!("Memory usage: {} bytes", dataset_generic.space_usage_byte());

    println!("\nConverting to compressed dataset...");
    let dataset_compressed =
        SparseDatasetCompressed::<u16, FixedU16Q>::from_dataset_f32(dataset_generic);

    println!("=== Compressed Dataset Info ===");
    println!("Number of Vectors: {}", dataset_compressed.len());
    println!("Number of Dimensions: {}", dataset_compressed.dim());
    println!(
        "Avg number of components: {:.2}",
        dataset_compressed.nnz() as f32 / dataset_compressed.len() as f32
    );
    println!("Total non-zero components: {}", dataset_compressed.nnz());
    println!(
        "Memory usage: {} bytes",
        dataset_compressed.space_usage_byte()
    );

    let output_path = match args.output_path {
        Some(path) => path,
        None => {
            eprintln!("Error: Output path is required. Use -o <path>");
            std::process::exit(1);
        }
    };

    let results: Vec<_> = queries
        .dataset_iter()
        .take(10)
        .map(|(query_components, query_values)| {
            dataset_compressed.search(
                query_components
                    .iter()
                    .zip(query_values)
                    .map(|(&c, &v)| (c, v)),
                k,
            )
        })
        .collect();

    let mut output_file = File::create(output_path).unwrap();

    let start = Instant::now();
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

    let duration = start.elapsed();
    println!("Total time taken: {:?}", duration);
    println!("Time per query: {:?}", duration / results.len() as u32);
    println!(
        "Average time per document: {:?}",
        duration / (results.len() as u32 * dataset_compressed.len() as u32)
    );
}
