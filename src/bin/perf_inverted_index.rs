use std::cmp;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use clap::Parser;
use num_traits::FromPrimitive;
use seismic::utils::read_from_path;
use seismic::InvertedIndex;
use vectorium::{
    ComponentType, Dataset as VDataset, Distance, DotProduct, DotVByteFixedU8Quantizer,
    PackedDataset, QueryVectorFor, SpaceUsage, SparseVector1D, Vector1D, VectorEncoder,
    read_seismic_format,
};

type ComponentFor<E> = <E as VectorEncoder>::OutputComponentType;
type QueryComponentFor<E> = <E as VectorEncoder>::QueryComponentType;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
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

    /// Value type: f16, bf16, f32, fixedu8, fixedu16, or dotvbyte.
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

macro_rules! match_component_value {
    ($component:expr, $value:expr, $macro_name:ident) => {
        use half::{bf16, f16};
        use seismic::{FixedU16Q, FixedU8Q};

        match ($component, $value) {
            ("u16", "f16") => {
                $macro_name!(u16, f16);
            }
            ("u16", "bf16") => {
                $macro_name!(u16, bf16);
            }
            ("u16", "f32") => {
                $macro_name!(u16, f32);
            }
            ("u16", "fixedu8") => {
                $macro_name!(u16, FixedU8Q);
            }
            ("u16", "fixedu16") => {
                $macro_name!(u16, FixedU16Q);
            }
            ("u32", "f16") => {
                $macro_name!(u32, f16);
            }
            ("u32", "bf16") => {
                $macro_name!(u32, bf16);
            }
            ("u32", "f32") => {
                $macro_name!(u32, f32);
            }
            ("u32", "fixedu8") => {
                $macro_name!(u32, FixedU8Q);
            }
            ("u32", "fixedu16") => {
                $macro_name!(u32, FixedU16Q);
            }
            _ => {
                eprintln!(
                    "Error: component-type must be either 'u16' or 'u32', value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'"
                );
                std::process::exit(1);
            }
        }
    };
}

pub fn run_performance_test_generic<S, E>(args: Args)
where
    S: VDataset<E> + Sync + SpaceUsage + serde::Serialize + serde::de::DeserializeOwned,
    E: VectorEncoder<QueryValueType = f32, Distance = DotProduct>,
    E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
    ComponentFor<E>:
        ComponentType + vectorium::ComponentType + FromPrimitive + SpaceUsage
        + serde::Serialize + serde::de::DeserializeOwned,
    QueryComponentFor<E>: ComponentType,
    SparseVector1D<ComponentFor<E>, f32, Vec<ComponentFor<E>>, Vec<f32>>: QueryVectorFor<E>,
{
    let index_path = args.index_file;
    let query_cut = args.query_cut;
    let heap_factor = args.heap_factor;
    let n_runs = args.n_runs;

    let nknn = args.n_knn;

    let inverted_index: InvertedIndex<S, E> =
        read_from_path(index_path.unwrap().as_str()).unwrap();

    let queries =
        read_seismic_format::<ComponentFor<E>, f32, DotProduct>(&args.query_file.unwrap()).unwrap();

    let n_queries = cmp::min(args.n_queries, VDataset::len(&queries));

    println!("Searching for top-{} results", args.k);
    println!("Number of evaluated queries: {n_queries}");
    println!(
        "Avg number of non-zero components: {}",
        VDataset::nnz(&queries) / VDataset::len(&queries)
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

        for (query_id, components_values) in VDataset::iter(&queries).take(n_queries).enumerate() {
            let q_components: Vec<_> = components_values.components_as_slice().to_vec();
            let q_values: Vec<_> = components_values.values_as_slice().to_vec();
            let query = SparseVector1D::new(q_components, q_values);
            let cur_results = inverted_index.search(
                &query,
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
        for (idx, scored) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{}\t{}\t{}",
                scored.vector,
                idx + 1,
                scored.distance.distance(),
            )
            .unwrap();
        }
    }
}

pub fn main() {
    let args = Args::parse();

    if args.value_type() == "dotvbyte" {
        if args.component_type() != "u16" {
            eprintln!("Error: value-type 'dotvbyte' is only supported with component-type 'u16'");
            std::process::exit(1);
        }

        println!("Using u16 component type with dotvbyte value type");
        run_performance_test_generic::<PackedDataset<DotVByteFixedU8Quantizer>, DotVByteFixedU8Quantizer>(
            args,
        );
        return;
    }

    macro_rules! run_performance_test_macro {
        ($C:ty, $V:ty) => {
            println!(
                "Using {} component type with {} value type",
                stringify!($C),
                stringify!($V)
            );
            run_performance_test_generic::<
                vectorium::PlainSparseDataset<$C, $V, vectorium::DotProduct>,
                vectorium::PlainSparseQuantizer<$C, $V, vectorium::DotProduct>,
            >(args);
        };
    }

    match_component_value!(
        args.component_type(),
        args.value_type(),
        run_performance_test_macro
    );
}
