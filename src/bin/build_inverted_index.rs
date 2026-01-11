use clap::Parser;
use half::{bf16, f16};
use num_traits::FromPrimitive;
use seismic::FixedU8Q;
use seismic::FixedU16Q;
use seismic::PlainInvertedIndex;
use seismic::ScalarInvertedIndex;
use seismic::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};
use seismic::utils::write_to_path;
use serde::Serialize;
use serde::de::DeserializeOwned;

use std::hash::Hash;
use std::time::Instant;

use vectorium::ComponentType;
use vectorium::{Dataset, DotProduct, SpaceUsage, read_seismic_format};

// clap does not support enums with associated values; keep CLI-only types in the bin.
#[derive(clap::ValueEnum, Default, Debug, Clone)]
#[clap(rename_all = "kebab-case")]
enum PruningStrategyClap {
    FixedSize,
    #[default]
    GlobalThreshold,
    CoiThreshold,
}

// clap does not support enums with associated values; keep CLI-only types in the bin.
#[derive(clap::ValueEnum, Default, Debug, Clone)]
#[clap(rename_all = "kebab-case")]
enum ClusteringAlgorithmClap {
    RandomKmeans,
    RandomKmeansInvertedIndex,
    #[default]
    RandomKmeansInvertedIndexApprox,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
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

    /// Component type: u16 (for component IDs up to 65535) or u32 (for larger component IDs)
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type: f16, bf16, f32, fixedu16, or fixedu8
    #[clap(long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

fn build_config(args: &Args) -> Configuration {
    let knn_config = KnnConfiguration::new(args.knn, args.knn_path.clone());

    let clustering = match args.clustering_algorithm {
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

    let pruning = match args.pruning_strategy {
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

    Configuration::default()
        .pruning_strategy(pruning)
        .blocking_strategy(BlockingStrategy::RandomKmeans {
            centroid_fraction: args.centroid_fraction,
            min_cluster_size: args.min_cluster_size,
            clustering_algorithm: clustering,
        })
        .summarization_strategy(SummarizationStrategy::EnergyPreserving {
            summary_energy: args.summary_energy,
        })
        .knn(knn_config)
}

fn build_base_index<C>(args: &Args) -> PlainInvertedIndex<C, f32>
where
    C: ComponentType
        + vectorium::ComponentType
        + SpaceUsage
        + Hash
        + FromPrimitive
        + Serialize
        + DeserializeOwned,
{
    let dataset =
        read_seismic_format::<C, f32, DotProduct>(args.input_file.as_ref().unwrap()).unwrap();

    println!("Number of Vectors: {}", dataset.len());
    println!("Number of Dimensions: {}", dataset.input_dim());
    println!(
        "Avg number of components: {:.2}",
        dataset.nnz() as f32 / dataset.len() as f32
    );

    let config = build_config(args);
    println!("\nBuilding the index...");
    println!("{:?}", config);

    PlainInvertedIndex::<C, f32>::build(dataset, config)
}

fn write_index<T: serde::Serialize>(index: T, output_file: &str, elapsed: Instant) {
    let path = output_file.to_string() + ".index.seismic";
    println!("Saving ... {}", path);
    let r = write_to_path(index, path.as_str());
    println!("{:?}", r);
    println!("Time to build {} secs", elapsed.elapsed().as_secs());
}

fn build_for_component<C>(args: &Args)
where
    C: ComponentType
        + vectorium::ComponentType
        + SpaceUsage
        + Hash
        + FromPrimitive
        + Serialize
        + DeserializeOwned,
{
    let time = Instant::now();
    let base_index = build_base_index::<C>(args);

    match args.value_type.as_str() {
        "f32" => write_index(base_index, args.output_file.as_ref().unwrap(), time),
        "f16" => write_index(
            ScalarInvertedIndex::<C, f32, f16>::convert_dataset_from(base_index),
            args.output_file.as_ref().unwrap(),
            time,
        ),
        "bf16" => write_index(
            ScalarInvertedIndex::<C, f32, bf16>::convert_dataset_from(base_index),
            args.output_file.as_ref().unwrap(),
            time,
        ),
        "fixedu8" => write_index(
            ScalarInvertedIndex::<C, f32, FixedU8Q>::convert_dataset_from(base_index),
            args.output_file.as_ref().unwrap(),
            time,
        ),
        "fixedu16" => write_index(
            ScalarInvertedIndex::<C, f32, FixedU16Q>::convert_dataset_from(base_index),
            args.output_file.as_ref().unwrap(),
            time,
        ),
        _ => {
            eprintln!("Error: value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args = Args::parse();
    match args.component_type.as_str() {
        "u16" => build_for_component::<u16>(&args),
        "u32" => build_for_component::<u32>(&args),
        _ => {
            eprintln!("Error: component-type must be either 'u16' or 'u32'");
            std::process::exit(1);
        }
    }
}
