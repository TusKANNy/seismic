// use clap::Parser;
// use indicatif::ProgressIterator;
// use seismic::*;
// use serde_json;
// use std::collections::HashMap;
// use std::fs::File;
// use std::io::Write;

// #[derive(Parser, Debug)]
// #[clap(author, version, about, long_about = None)]
// struct Args {
//     /// The path of the input file
//     #[clap(short, long, value_parser)]
//     input_file: Option<String>,

//     /// The output prefix for the inverted index files
//     #[clap(short, long, value_parser, default_value = "inverted_index")]
//     output_prefix: String,
// }

pub fn main() {
    // let args = Args::parse();

    // let input_file = args.input_file.expect("Input file is required");
    // let output_prefix = args.output_prefix;

    // let dataset = SparseDataset::<u16, f32>::from_dataset_f32(
    //     SparseDatasetMut::<u16, f32>::read_bin_file(&input_file).unwrap(),
    // );

    // let mut inverted_index: HashMap<u16, Vec<f32>> = HashMap::new();

    // for document in dataset.iter().progress_count(dataset.len() as u64) {
    //     let (components, values) = document;
    //     for (component, value) in components.iter().zip(values.iter()) {
    //         inverted_index
    //             .entry(*component)
    //             .or_insert_with(Vec::new)
    //             .push(*value);
    //     }
    // }

    // // Sort components for consistent output
    // let mut components: Vec<u16> = inverted_index.keys().cloned().collect();
    // components.sort();

    // // Build arrays: component counts and all values
    // let mut component_counts: Vec<u32> = Vec::new();
    // let mut all_values: Vec<f32> = Vec::new();

    // println!("Building arrays...");
    // for component in components.iter().progress() {
    //     let values = &inverted_index[component];
    //     component_counts.push(values.len() as u32);
    //     all_values.extend_from_slice(values);
    // }

    // println!("Computing the quantization bins");

    // let mut quantization_bins =
    //     HashMap::<u16, (f32, f32, f32, f32)>::with_capacity(components.len());
    // let nbits = 8.0;
    // for &component in components.iter().progress_count(components.len() as u64) {
    //     let s = &inverted_index[&component];
    //     let min = s.iter().cloned().fold(f32::INFINITY, f32::min);
    //     let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    //     // Calculate scale using the formula: scale = (max/min)^(1/2^nbits)
    //     let scale = (max / min).powf(1.0 / (2.0_f32.powf(nbits)));
    //     let mean = s.iter().sum::<f32>() / s.len() as f32;
    //     quantization_bins.insert(component, (min, max, scale, mean));
    // }

    // // Save quantization bins as JSON
    // let bins_file = format!("{}_quantization_bins.json", output_prefix);
    // save_quantization_bins_json(&quantization_bins, &bins_file)
    //     .expect("Failed to save quantization bins");

    // // Save component counts as binary (NumPy compatible)
    // let counts_file = format!("{}_counts.npy", output_prefix);
    // save_npy_u32(&component_counts, &counts_file).expect("Failed to save counts");

    // // Save all values as binary (NumPy compatible)
    // let values_file = format!("{}_values.npy", output_prefix);
    // save_npy_f32(&all_values, &values_file).expect("Failed to save values");

    // println!(
    //     "Saved inverted index with {} components:",
    //     inverted_index.len()
    // );
    // println!(
    //     "  - {}_counts.npy ({} component counts)",
    //     output_prefix,
    //     component_counts.len()
    // );
    // println!(
    //     "  - {}_values.npy ({} values)",
    //     output_prefix,
    //     all_values.len()
    // );
    // println!(
    //     "  - {}_quantization_bins.json ({} quantization bins)",
    //     output_prefix,
    //     quantization_bins.len()
    // );

    // println!(
    //     "Quantization bins computed for {} components",
    //     quantization_bins.len()
    // );
}

// fn save_npy_u32(data: &[u32], filename: &str) -> std::io::Result<()> {
//     use std::io::Write;
//     let mut file = File::create(filename)?;

//     // NumPy header for u32 array
//     let header = format!(
//         "{{'descr': '<u4', 'fortran_order': False, 'shape': ({},), }}",
//         data.len()
//     );
//     let header_len = header.len();
//     let padding = (16 - (header_len + 1) % 16) % 16;
//     let padded_header = format!("{}{}", header, " ".repeat(padding));

//     // Magic number + version
//     file.write_all(b"\x93NUMPY")?;
//     file.write_all(&[1, 0])?; // version 1.0
//     file.write_all(&(padded_header.len() as u16).to_le_bytes())?;
//     file.write_all(padded_header.as_bytes())?;

//     // Data
//     for &value in data {
//         file.write_all(&value.to_le_bytes())?;
//     }

//     Ok(())
// }

// fn save_npy_f32(data: &[f32], filename: &str) -> std::io::Result<()> {
//     use std::io::Write;
//     let mut file = File::create(filename)?;

//     // NumPy header for f32 array
//     let header = format!(
//         "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
//         data.len()
//     );
//     let header_len = header.len();
//     let padding = (16 - (header_len + 1) % 16) % 16;
//     let padded_header = format!("{}{}", header, " ".repeat(padding));

//     // Magic number + version
//     file.write_all(b"\x93NUMPY")?;
//     file.write_all(&[1, 0])?; // version 1.0
//     file.write_all(&(padded_header.len() as u16).to_le_bytes())?;
//     file.write_all(padded_header.as_bytes())?;

//     // Data
//     for &value in data {
//         file.write_all(&value.to_le_bytes())?;
//     }

//     Ok(())
// }

// fn save_quantization_bins_json(
//     quantization_bins: &HashMap<u16, (f32, f32, f32, f32)>,
//     filename: &str,
// ) -> std::io::Result<()> {
//     let mut file = File::create(filename)?;

//     // Create a JSON object with component IDs as keys and bins as values
//     let mut json_data = serde_json::Map::new();

//     for (&component_id, &(min, max, scale, mean)) in quantization_bins.iter() {
//         let bin_data = serde_json::json!({
//             "min": min,
//             "max": max,
//             "scale": scale,
//             "mean": mean
//         });
//         json_data.insert(component_id.to_string(), bin_data);
//     }

//     let json_object = serde_json::Value::Object(json_data);
//     let json_string = serde_json::to_string_pretty(&json_object)?;
//     file.write_all(json_string.as_bytes())?;

//     Ok(())
// }
