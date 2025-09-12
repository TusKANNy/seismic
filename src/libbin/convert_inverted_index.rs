use clap::Parser;
use std::time::Instant;

use crate::utils::{read_from_path, write_to_path};
use crate::{FromDatasetGenericF32, InvertedIndex, SparseDataset, SparseDatasetTrait};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// The path of the input file
    #[clap(short, long, value_parser)]
    index_file: Option<String>,

    /// The path of the output file. The extension will encode the values of the building parameters.
    #[clap(short, long, value_parser)]
    pub output_file: Option<String>,

    /// Component type: u16 (for component IDs up to 65535) or u32 (for larger component IDs)
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type: f16, bf16, f32, fixedu16, or fixedu8
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

pub fn convert_index_from_f32<S>(args: Args)
where
    S: SparseDatasetTrait
        + FromDatasetGenericF32<SparseDataset<S::Component, f32>>
        + Sync
        + serde::Serialize
        + serde::de::DeserializeOwned,
    S::Component: serde::Serialize + serde::de::DeserializeOwned,
{
    println!("Loading inverted index...");

    let inverted_index: InvertedIndex<SparseDataset<S::Component, f32>> =
        read_from_path(&args.index_file.unwrap()).unwrap();

    println!("Number of Vectors: {}", inverted_index.len());
    println!("Number of Dimensions: {}", inverted_index.dim());

    println!(
        "Avg number of components: {:.2}",
        inverted_index.nnz() as f32 / inverted_index.len() as f32
    );

    let time = Instant::now();

    println!("Converting the inverted index...");
    let inverted_index_partitioned = InvertedIndex::<S>::from_inverted_index(inverted_index);

    let elapsed = time.elapsed();
    println!(
        "Time to convert {} secs (before serializing)",
        elapsed.as_secs()
    );

    let path = args.output_file.unwrap() + ".index.seismic";

    println!("Saving ... {}", path);
    let r = write_to_path(inverted_index_partitioned, path.as_str());

    println!("{:?}", r);

    let elapsed = time.elapsed();
    println!("Time to build {} secs", elapsed.as_secs());
}
