use clap::Parser;
use seismic::InvertedIndex;
use seismic::FixedU8Q;
use seismic::stream_vbyte_dataset::dataset::SparseDatasetStreamVbyte;
use seismic::sparse_dataset::{SparseDatasetMut, SparseDatasetTrait};

use seismic::utils::read_from_path;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::prelude::IndexedRandom;
use std::time::Instant;
use std::cmp;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: Option<String>,

    /// The query file.
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// The number of queries to evaluate.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10000)]
    n_queries: usize,

    /// The number of docs to dot product with.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 1)]
    n_docs: usize,

    /// Component type: u16 (for component IDs up to 65535) or u32 (for larger component IDs)
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type: f16, bf16, f32, fixedu8, or fixedu16.
    #[clap(long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

impl Args {
    pub fn component_type(&self) -> &str {
        &self.component_type
    }

    pub fn value_type(&self) -> &str {
        &self.value_type
    }
}

fn main() {
    let args = Args::parse();

    let index_path = args.index_file;
    let n_docs = args.n_docs;

    let inverted_index: InvertedIndex<SparseDatasetStreamVbyte<FixedU8Q>> = read_from_path(index_path.unwrap().as_str()).unwrap();

    let queries =
        SparseDatasetMut::<u16, f32>::read_bin_file(&args.query_file.unwrap()).unwrap();

    let n_queries = cmp::min(args.n_queries, queries.len());

    println!("Dot products with {} documents", n_docs);

    println!("Number of documents: {}", inverted_index.len());
    println!(
        "Avg number of non-zero components: {}",
        inverted_index.nnz() / inverted_index.len()
    );

    // Random selection of document ids
    let mut rng = rand::thread_rng();
    let doc_ids: Vec<usize> = (0..inverted_index.len())
        .collect::<Vec<_>>()
        .choose_multiple(&mut rng, n_docs)
        .copied()
        .collect();
    // For each selected doc id store the (offset, len) so we can call dot_product_from_offset
    let doc_ranges: Vec<(usize, usize)> = doc_ids
        .iter()
        .map(|&doc_id| {
            let range = inverted_index.dataset().offset_range(doc_id);
            (range.start, range.len())
        })
        .collect();

    println!("Running dot products...");
    let mut sum = 0.0;
    let time = Instant::now();

    for (q_components, q_values) in queries.dataset_iter().take(n_queries) {
        // Prepare the query once for this dataset implementation
        let prepared_query = inverted_index
            .dataset()
            .prepare_query::<u16, f32>(q_components.iter().copied().zip(q_values.iter().copied()));

        // Compute dot-product of the prepared query against each selected document
        for &(offset, len) in doc_ranges.iter() {
            let score = inverted_index
                .dataset()
                .dot_product_from_offset::<u16, f32>(&prepared_query, offset, len);
            sum += score;
        }
    }

    let elapsed = time.elapsed();
    println!(
        "Time {} nanosecs per dot product",
        elapsed.as_nanos() / (n_queries * n_docs) as u128
    );

    println!("Sum of all scores: {}", sum);
}
