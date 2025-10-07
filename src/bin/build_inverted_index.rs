use half::bf16;
use half::f16;
use seismic::FixedU8Q;
use seismic::FixedU16Q;
use seismic::inverted_index::{
    BlockingStrategy, ClusteringAlgorithm, ClusteringAlgorithmClap, Configuration,
    KnnConfiguration, PruningStrategy, PruningStrategyClap, SummarizationStrategy,
};
use seismic::utils::write_to_path;
use seismic::*;

use clap::Parser;
use std::time::Instant;

// TODO:
// - add control to the Rayon's number of threads

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the input file
    #[clap(short, long, value_parser)]
    input_file: Option<String>,

    /// The path of the output file. The extension will encode the values of the building parameters.
    #[clap(short, long, value_parser)]
    output_file: Option<String>,

    /// The number of postings to be selected in each posting list.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 6000)]
    n_postings: usize,

    /// Block size in the fixed size blocking.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    block_size: usize,

    /// Regulates the number of centroids built for each posting list. The number of centroids is at most the specified fraction of the posting list length.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.1)]
    centroid_fraction: f32,

    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 0.5)]
    summary_energy: f32,

    /// Selects the clustering algorithm used to cluster postings in each posting list
    #[clap(long, value_parser)]
    // #[arg(default_value_t = ClusteringAlgorithmClap::default())]
    clustering_algorithm: ClusteringAlgorithmClap,

    #[clap(long, value_parser)]
    pruning_strategy: PruningStrategyClap,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.005)]
    kmeans_pruning_factor: f32,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 15)]
    kmeans_doc_cut: usize,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 2)]
    min_cluster_size: usize,

    /// Regulates the fraction of L1 mass preserved by the COI pruning strategy.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 0.15)]
    alpha: f32,

    /// Regulates the largest lenght of a posting list as a factor of n_postings parameter.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 1.5)]
    max_fraction: f32,

    /// Says how many neighbors to include for each vector of the dataset.
    /// These neighbors are used to improve the accuracy of the reported results.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    knn: usize,

    /// Path to the file of precomputed nearest neighbors.
    #[clap(long, value_parser)]
    knn_path: Option<String>,

    /// Number of documents per chunk in the batched indexing mode.
    #[clap(long, value_parser)]
    batched_indexing: Option<usize>,

    /// Component type: u16 (for component IDs up to 65535) or u32 (for larger component IDs)
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type: f16, bf16, f32, fixedu16, or fixedu8
    #[clap(long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

fn build_index_generic<C, D>(args: Args)
where
    C: ComponentType + serde::Serialize + for<'de> serde::Deserialize<'de>,
    D: ValueType + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let dataset = SparseDataset::<C, D>::from_dataset_f32(
        SparseDatasetMut::<C, f32>::read_bin_file(&args.input_file.unwrap()).unwrap(),
    );

    println!("Number of vectors: {}", dataset.len());
    println!("Number of dimensions: {}", dataset.dim());

    println!(
        "Avg. number of components: {:.2}",
        dataset.nnz() as f32 / dataset.len() as f32
    );

    let time = Instant::now();

    let knn_config = KnnConfiguration::new(args.knn, args.knn_path.clone());

    let my_clustering_algorithm = match args.clustering_algorithm {
        ClusteringAlgorithmClap::RandomKmeansInvertedIndexApprox => {
            ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                doc_cut: args.kmeans_doc_cut,
            }
        }
        ClusteringAlgorithmClap::RandomKmeansInvertedIndex => {
            ClusteringAlgorithm::RandomKmeansInvertedIndex {
                pruning_factor: args.kmeans_pruning_factor,
                doc_cut: args.kmeans_doc_cut,
            }
        }
        ClusteringAlgorithmClap::RandomKmeans => ClusteringAlgorithm::RandomKmeans {},
    };

    let my_pruning_strategy = match args.pruning_strategy {
        PruningStrategyClap::FixedSize => PruningStrategy::FixedSize {
            n_postings: args.n_postings,
        },
        PruningStrategyClap::GlobalThreshold => PruningStrategy::GlobalThreshold {
            n_postings: args.n_postings,
            max_fraction: args.max_fraction,
        },
        PruningStrategyClap::CoiThreshold => PruningStrategy::CoiThreshold {
            alpha: args.alpha,
            n_postings: args.n_postings,
        },
    };

    let config = Configuration::default()
        .pruning_strategy(my_pruning_strategy)
        .blocking_strategy(BlockingStrategy::RandomKmeans {
            centroid_fraction: args.centroid_fraction,
            min_cluster_size: args.min_cluster_size,
            clustering_algorithm: my_clustering_algorithm,
        })
        .summarization_strategy(SummarizationStrategy::EnergyPreserving {
            summary_energy: args.summary_energy,
        })
        .knn(knn_config)
        .batched_indexing(args.batched_indexing);

    println!("\nBuilding the index...");
    println!("{:?}", config);

    let inverted_index = InvertedIndex::build(dataset, config);

    let elapsed = time.elapsed();
    println!(
        "Time to build {} secs (before serializing)",
        elapsed.as_secs()
    );

    let path = args.output_file.unwrap() + ".index.seismic";

    println!("Index saved to: {}", path);
    let _ = write_to_path(inverted_index, path.as_str());

    //println!("{:?}", r);

    let elapsed = time.elapsed();
    println!("Time to build {} secs (with serialization)", elapsed.as_secs());
}

pub fn main() {
    let args = Args::parse();

    match (args.component_type.as_str(), args.value_type.as_str()) {
        ("u16", "f16") => {
            println!("Using u16 component type with f16 value type");
            build_index_generic::<u16, f16>(args);
        }
        ("u16", "bf16") => {
            println!("Using u16 component type with bf16 value type");
            build_index_generic::<u16, bf16>(args);
        }
        ("u16", "f32") => {
            println!("Using u16 component type with f32 value type");
            build_index_generic::<u16, f32>(args);
        }
        ("u16", "fixedu16") => {
            println!("Using u16 component type with fixedu16 value type");
            build_index_generic::<u16, FixedU16Q>(args);
        }
        ("u16", "fixedu8") => {
            println!("Using u16 component type with fixedu8 value type");
            build_index_generic::<u16, FixedU8Q>(args);
        }
        ("u32", "f16") => {
            println!("Using u32 component type with f16 value type");
            build_index_generic::<u32, f16>(args);
        }
        ("u32", "bf16") => {
            println!("Using u32 component type with bf16 value type");
            build_index_generic::<u32, bf16>(args);
        }
        ("u32", "f32") => {
            println!("Using u32 component type with f32 value type");
            build_index_generic::<u32, f32>(args);
        }
        ("u32", "fixedu16") => {
            println!("Using u32 component type with fixedu16 value type");
            build_index_generic::<u32, FixedU16Q>(args);
        }
        ("u32", "fixedu8") => {
            println!("Using u32 component type with fixedu8 value type");
            build_index_generic::<u32, FixedU8Q>(args);
        }
        _ => {
            eprintln!(
                "Error: component-type must be either 'u16' or 'u32', value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'"
            );
            std::process::exit(1);
        }
    }
}
