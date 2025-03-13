use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    time::Instant,
};

use crate::json_utils::{extract_jsonl, JsonFormat};

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{
    inverted_index::Configuration, inverted_index::Knn, DataType, InvertedIndex, SpaceUsage,
    SparseDataset, SparseDatasetMut,
};

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<T>
where
    T: DataType,
{
    inverted_index: InvertedIndex<T>,
    document_mapping: Option<Box<[String]>>,
    token_to_id_map: HashMap<String, usize>,
}

impl<T> SpaceUsage for SeismicIndex<T>
where
    T: DataType,
{
    fn space_usage_byte(&self) -> usize {
        //TODO: add the SpaceUsage of document_mapping and token_to_id_map
        self.inverted_index.space_usage_byte()
    }
}

impl<T> SeismicIndex<T>
where
    T: PartialOrd + DataType,
{
    pub fn new(
        dataset: SparseDataset<T>,
        config: Configuration,
        document_mapping: Option<Vec<String>>,
        token_to_id_map: HashMap<String, usize>,
    ) -> Self
    where
        T: DataType + PartialOrd,
    {
        let inverted_index = InvertedIndex::build(dataset, config);

        Self {
            inverted_index,
            document_mapping: Some(document_mapping.unwrap().into_boxed_slice()),
            token_to_id_map,
        }
    }

    pub fn remap_doc_ids(
        &self,
        plain_results: Vec<(f32, usize)>,
        query_id: &String,
    ) -> Vec<(String, f32, String)> {
        let remapped_results: Vec<(String, f32, String)> = match &self.document_mapping {
            Some(mapp) => plain_results
                .iter()
                .map(|(distance, doc_id)| (query_id.clone(), *distance, mapp[*doc_id].clone()))
                .collect(),
            None => plain_results
                .iter()
                .map(|(distance, doc_id)| (query_id.clone(), *distance, doc_id.to_string()))
                .collect(),
        };

        remapped_results
    }

    pub fn search_raw(
        &self,
        query_components_original: &[String],
        query_values: &[f32],
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<(f32, usize)> {
        let (filtered_query_components, filtered_query_values): (Vec<_>, Vec<_>) =
            query_components_original
                .iter()
                .zip(query_values)
                .filter(|(qc, _)| self.token_to_id_map.contains_key(*qc))
                .map(|(qc, &qv)| (self.token_to_id_map[qc] as u16, qv))
                .unzip();

        let results = self.inverted_index.search(
            &filtered_query_components,
            &filtered_query_values,
            k,
            query_cut,
            heap_factor,
            n_knn,
            first_sorted,
        );

        results
    }

    pub fn search(
        &self,
        query_id: &String,
        query_components_original: &[String],
        query_values: &[f32],
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<(String, f32, String)> {
        let results = self.search_raw(
            query_components_original,
            query_values,
            k,
            query_cut,
            heap_factor,
            n_knn,
            first_sorted,
        );

        self.remap_doc_ids(results, query_id) // return the documents remapped
    }

    pub fn process_data(
        reader: BufReader<impl std::io::Read>,
        row_count: usize,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> (SparseDataset<T>, Vec<String>, HashMap<String, usize>) {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let mut token_to_id_mapping = HashMap::<String, usize>::with_capacity(30_000);
        let mut converted_data = SparseDatasetMut::<T>::default();

        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values) = extract_jsonl::<T>(x.unwrap());
            doc_id_mapping.push(doc_id);

            let ids: Vec<_> = match &input_token_to_id_map {
                None => {
                    for token in tokens.iter() {
                        if !token_to_id_mapping.contains_key(token) {
                            token_to_id_mapping
                                .insert(token.to_string(), token_to_id_mapping.len());
                        }
                    }
                    tokens
                        .iter()
                        .map(|t| token_to_id_mapping[t] as u16)
                        .collect()
                }
                Some(valid_mapping) => tokens.iter().map(|t| valid_mapping[t] as u16).collect(),
            };

            let converted_vector: Vec<(u16, T)> = ids
                .into_iter()
                .zip(values)
                .sorted_by(|(a, _), (b, _)| a.partial_cmp(&b).unwrap())
                .collect();

            converted_data.push_pairs(&converted_vector[..]);
        }

        let final_data: SparseDataset<T> = SparseDataset::<T>::from(converted_data);

        (final_data, doc_id_mapping, token_to_id_mapping)
    }

    pub fn from_file(
        file_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Result<Self, io::Error> {
        if file_path.ends_with(".jsonl") {
            Ok(SeismicIndex::from_json(
                file_path,
                config,
                input_token_to_id_map,
            ))
        } else if file_path.ends_with(".tar.gz") {
            Ok(SeismicIndex::from_tar(
                file_path,
                config,
                input_token_to_id_map,
            ))
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unsupported file type. Supported files: .jsonl, .tar.gz",
            ))
        }
    }

    pub fn from_json(
        json_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self {
        println!("Reading the collection...");
        let start = Instant::now();

        //read the file and count rows
        let f = File::open(json_path).unwrap();
        let reader = io::BufReader::new(f);
        let row_count = reader.lines().count();

        println!("Number of rows: {}", row_count);

        let f = File::open(json_path).expect(&format!("Unable to open {}", json_path));
        let reader = BufReader::new(f);

        let (final_data, doc_id_mapping, token_to_id_mapping) =
            Self::process_data(reader, row_count, input_token_to_id_map);

        println!(
            "Elapsed time to read the collection: {:} secs",
            start.elapsed().as_secs()
        );

        Self::new(
            final_data,
            config,
            Some(doc_id_mapping),
            token_to_id_mapping,
        )
    }

    #[allow(unused_variables)]
    pub fn from_tar(
        tar_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self {
        println!("Reading the collection...");
        let start = Instant::now();

        //decompress gz, extract first file (json) of the archive
        //read the file and count rows
        let tar_gz_file = File::open(tar_path).expect(&format!("Unable to open {}", tar_path));
        let gz_decoder = GzDecoder::new(tar_gz_file);
        let mut archive = Archive::new(gz_decoder);
        let json_entry = archive.entries().unwrap().next().unwrap().unwrap();
        let reader = BufReader::new(json_entry);
        let row_count = reader.lines().count();

        println!("Number of rows: {}", row_count);

        //Deserialize json
        let tar_gz_file = File::open(tar_path).expect(&format!("Unable to open {}", tar_path));
        let gz_decoder = GzDecoder::new(tar_gz_file);
        let mut archive = Archive::new(gz_decoder);
        let json_entry = archive.entries().unwrap().next().unwrap().unwrap();
        let reader = BufReader::new(json_entry);

        let (final_data, doc_id_mapping, token_to_id_mapping) =
            Self::process_data(reader, row_count, input_token_to_id_map);

        println!(
            "Elapsed time to read the collection: {:} secs",
            start.elapsed().as_secs()
        );

        Self::new(
            final_data,
            config,
            Some(doc_id_mapping),
            token_to_id_mapping,
        )
    }

    pub fn print_space_usage_byte(&self) {
        self.inverted_index.print_space_usage_byte();
    }

    pub fn dim(&self) -> usize {
        self.inverted_index.dim()
    }

    pub fn nnz(&self) -> usize {
        self.inverted_index.nnz()
    }

    pub fn len(&self) -> usize {
        self.inverted_index.len()
    }

    pub fn inverted_index(&self) -> &InvertedIndex<T> {
        &self.inverted_index
    }

    pub fn dataset(&self) -> &SparseDataset<T> {
        &self.inverted_index.dataset()
    }

    pub fn add_knn(&mut self, knn: Knn) {
        self.inverted_index.add_knn(knn);
    }

    pub fn knn_len(&self) -> usize {
        self.inverted_index.knn_len()
    }
}
