use clap::Parser;
use compressed_intvec::prelude::VariableCodecSpec;
use indicatif::ProgressIterator;
use seismic::{
    FixedU16Q, SpaceUsage, SparseDatasetTrait, compressed_dataset::SparseDatasetCompressed,
    sparse_dataset::SparseDatasetMut,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Debug)]
struct PerformanceMetrics {
    dataset_name: String,
    codec_type: String,
    codec_args: String,
    num_documents: usize,
    num_dimensions: usize,
    total_nnz: usize,
    avg_components_per_doc: f32,
    original_memory_bytes: usize,
    compressed_memory_bytes: usize,
    component_memory_bytes: usize,
    compression_ratio: f64,
    num_queries: usize,
    k_results: usize,
    total_search_time_ms: f64,
    avg_time_per_query_ms: f64,
    avg_time_per_document_ns: f64,
}

#[derive(clap::ValueEnum, Default, Debug, Clone)]
pub enum CompressionAlgorithmClap {
    #[default]
    Zeta,
    Gamma,
    Delta,
    VByteLe,
    VByteBe,
}

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

    /// Log path for TSV metrics file
    #[clap(short, long, value_parser)]
    log_path: Option<String>,

    /// The codec to use for compression (e.g., "zeta", "gamma", "delta")
    #[clap(short, long, value_parser)]
    #[arg(default_value = "zeta")]
    codec: CompressionAlgorithmClap,

    /// The number of queries to use for evaluation
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    n_queries: usize,

    /// K value for Zeta codec (only if codec is "zeta")
    #[clap(short, long, value_parser)]
    zeta_k: Option<u64>,
}

fn codec_args_to_string(codec: &VariableCodecSpec) -> String {
    match codec {
        VariableCodecSpec::Zeta { k } => match k {
            Some(value) => format!("k={}", value),
            _ => "k=auto".to_string(),
        },
        _ => "none".to_string(), // For codecs without parameters like Gamma, Delta, VByte, etc.
    }
}

fn write_metrics_to_tsv(metrics: &PerformanceMetrics, log_path: &str) -> std::io::Result<()> {
    let file_exists = std::path::Path::new(log_path).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    // Write header if file is new
    if !file_exists {
        writeln!(
            file,
            "dataset_name\tcodec_type\tcodec_args\tnum_documents\tnum_dimensions\ttotal_nnz\tavg_components_per_doc\toriginal_memory_bytes\tcompressed_memory_bytes\tcomponent_memory_bytes\tcompression_ratio\tnum_queries\tk_results\ttotal_search_time_ms\tavg_time_per_query_ms\tavg_time_per_document_ns"
        )?;
    }

    // Write data
    writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}",
        metrics.dataset_name,
        metrics.codec_type,
        metrics.codec_args,
        metrics.num_documents,
        metrics.num_dimensions,
        metrics.total_nnz,
        metrics.avg_components_per_doc,
        metrics.original_memory_bytes,
        metrics.compressed_memory_bytes,
        metrics.component_memory_bytes,
        metrics.compression_ratio,
        metrics.num_queries,
        metrics.k_results,
        metrics.total_search_time_ms,
        metrics.avg_time_per_query_ms,
        metrics.avg_time_per_document_ns
    )?;

    Ok(())
}

pub fn main() {
    let args = Args::parse();

    let input_file = match args.input_file {
        Some(file) => file,
        _ => {
            eprintln!("Error: Input file is required. Use -i <path>");
            std::process::exit(1);
        }
    };

    // === Write results to file ===
    let output_path = match args.output_path {
        Some(path) => path,
        _ => {
            eprintln!("Error: Output path is required. Use -o <path>");
            std::process::exit(1);
        }
    };

    let log_path = match args.log_path {
        Some(path) => path,
        _ => {
            eprintln!("Error: Log path is required. Use -l <path>");
            std::process::exit(1);
        }
    };

    let n_queries = args.n_queries;

    let codec = match args.codec {
        CompressionAlgorithmClap::Zeta => VariableCodecSpec::Zeta { k: args.zeta_k },
        CompressionAlgorithmClap::Gamma => VariableCodecSpec::Gamma,
        CompressionAlgorithmClap::Delta => VariableCodecSpec::Delta,
        CompressionAlgorithmClap::VByteLe => VariableCodecSpec::VByteLe,
        CompressionAlgorithmClap::VByteBe => VariableCodecSpec::VByteBe,
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

    println!("\n=== Generic Dataset Info ===");
    println!("Number of Vectors: {}", dataset_generic.len());
    println!("Number of Dimensions: {}", dataset_generic.dim());
    println!(
        "Avg number of components: {:.2}",
        dataset_generic.nnz() as f32 / dataset_generic.len() as f32
    );
    println!("Total non-zero components: {}", dataset_generic.nnz());
    let generic_memory_usage = dataset_generic.space_usage_byte();
    println!("Memory usage: {} bytes", generic_memory_usage);

    println!(
        "\nConverting to compressed dataset using {:?} codec...",
        codec
    );
    let dataset_compressed = SparseDatasetCompressed::<u16, FixedU16Q>::from_dataset_f32_with_codec(
        dataset_generic,
        codec,
    );

    println!("\n=== Compressed Dataset Info ===");
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

    // === Compressed Dataset Performance Test ===
    println!("\n=== Compressed Dataset Search Performance ===");
    let start = Instant::now();
    let results_compressed: Vec<_> = queries
        .dataset_iter()
        .take(n_queries)
        .progress_count(n_queries as u64)
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

    let duration_compressed = start.elapsed();

    println!(
        "Total time taken with compressed dataset: {:?}",
        duration_compressed
    );
    println!(
        "Time per query with compressed dataset: {:?}",
        duration_compressed / results_compressed.len() as u32
    );
    println!(
        "Average time per document with compressed dataset: {:?}",
        duration_compressed / (results_compressed.len() as u32 * dataset_compressed.len() as u32)
    );

    println!(
        "Compressed Dataset - Time per query: {:?}",
        duration_compressed / results_compressed.len() as u32
    );

    let memory_ratio = generic_memory_usage as f64 / dataset_compressed.space_usage_byte() as f64;
    println!(
        "Memory compression ratio: {:.2}x (FP uses {:.2}x more memory)",
        memory_ratio, memory_ratio
    );

    let component_memory_usage = dataset_compressed.space_usage_byte_components();

    // Collect metrics for TSV logging
    let metrics = PerformanceMetrics {
        dataset_name: input_file.clone(),
        codec_type: format!("{:?}", codec)
            .split(' ')
            .next()
            .unwrap()
            .to_string(),
        codec_args: codec_args_to_string(&codec),
        num_documents: dataset_compressed.len(),
        num_dimensions: dataset_compressed.dim(),
        total_nnz: dataset_compressed.nnz(),
        avg_components_per_doc: dataset_compressed.nnz() as f32 / dataset_compressed.len() as f32,
        original_memory_bytes: generic_memory_usage,
        compressed_memory_bytes: dataset_compressed.space_usage_byte(),
        component_memory_bytes: component_memory_usage,
        compression_ratio: memory_ratio,
        num_queries: results_compressed.len(),
        k_results: k,
        total_search_time_ms: duration_compressed.as_secs_f64() * 1000.0,
        avg_time_per_query_ms: (duration_compressed.as_micros() as f64)
            / results_compressed.len() as f64,
        avg_time_per_document_ns: (duration_compressed.as_nanos() as f64)
            / (results_compressed.len() as f64 * dataset_compressed.len() as f64),
    };

    // Write metrics to TSV log
    if let Err(e) = write_metrics_to_tsv(&metrics, &log_path) {
        eprintln!(
            "Warning: Failed to write metrics to log file {}: {}",
            log_path, e
        );
    } else {
        println!("Metrics logged to: {}", log_path);
    }

    let mut output_file = File::create(output_path).unwrap();

    // Write results from compressed dataset search (not timed separately)
    for (query_id, result) in results_compressed.iter().enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{}\t{}\t{}\t{}",
                query_id,
                doc_id,
                idx + 1,
                score
            )
            .unwrap();
        }
    }

    println!("\nResults written to file successfully.");
}
