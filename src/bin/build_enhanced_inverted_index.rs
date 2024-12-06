use seismic::inverted_index::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy, ClusteringAlgorithmClap
};
use seismic::SeismicIndex;

use half::f16;
use std::fs;

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

    /// The path of the output file. The extension will encode the values of thebuilding parameters.
    #[clap(short, long, value_parser)]
    output_file: Option<String>,

    /// The number of postings to be selected in each posting list.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 6000)]
    n_postings: usize,

    /// Block size in the fixed size blockin
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    block_size: usize,

    /// Regulates the number of centroids built for each posting list. The number of centroids is at most the fraction of the posting list lenght.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.1)]
    centroid_fraction: f32,

    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 0.5)]
    summary_energy: f32,
    
    #[clap(long, value_parser)]
   // #[arg(default_value_t = ClusteringAlgorithmClap::default())]
    clustering_algorithm: ClusteringAlgorithmClap,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.005)]
    kmeans_pruning_factor: f32,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 15)]
    kmeans_doc_cut: usize,

    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 2)]
    min_cluster_size: usize,

    /// Says how many neighbors to include for each vector of the dataset.
    /// These neighbors are used to improve the accuracy of the reported results.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    knn: usize,

    /// Path to the file of precomputed neareast neighbors.
    #[clap(long, value_parser)]
    knn_path: Option<String>,
    
    /// Number of documents per chunk in the batched indexing mode.
    #[clap(short, long, value_parser)]
    batched_indexing: Option<usize>,
}

pub fn main() {
    let args = Args::parse();

    let time = Instant::now();

    let knn_config = KnnConfiguration::new(args.knn, args.knn_path);

    let my_clustering_algorithm = match args.clustering_algorithm {
        ClusteringAlgorithmClap::RandomKmeansInvertedIndexApprox =>
        ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
            doc_cut: args.kmeans_doc_cut,
        }, 
        ClusteringAlgorithmClap::RandomKmeansInvertedIndex =>
        ClusteringAlgorithm::RandomKmeansInvertedIndex {
            pruning_factor: args.kmeans_pruning_factor,
            doc_cut: args.kmeans_doc_cut,
        },
        ClusteringAlgorithmClap::RandomKmeans => ClusteringAlgorithm::RandomKmeans {},
    };

    let config = Configuration::default()
        .pruning_strategy(PruningStrategy::GlobalThreshold {
            n_postings: args.n_postings,
            max_fraction: 1.5,
        })
        .blocking_strategy(BlockingStrategy::RandomKmeans {
            centroid_fraction: args.centroid_fraction,
            min_cluster_size: args.min_cluster_size,
            clustering_algorithm: my_clustering_algorithm,
        })
        .summarization_strategy(SummarizationStrategy::EnergyPreserving {
            summary_energy: args.summary_energy,
        })
        .knn(knn_config);
    println!("\nBuilding the index...");
    println!("{:?}", config);

    //    let inverted_index = InvertedIndexWrapper::new(dataset, config, None, None);
    let collection_path = args.input_file.unwrap();

    let index = SeismicIndex::<f16>::from_json(&collection_path, config, None);

    let elapsed = time.elapsed();
    println!(
        "Time to build {} secs (before serializing)",
        elapsed.as_secs()
    );

    let serialized = bincode::serialize(&index).unwrap();

    let path = args.output_file.unwrap() + ".index.seismic";

    println!("Saving ... {}", path);
    let r = fs::write(path, serialized);
    println!("{:?}", r);

    let elapsed = time.elapsed();
    println!("Time to build {} secs", elapsed.as_secs());
}