use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    time::Instant,
};

use crate::{
    ComponentType, FromDatasetGenericF32,
    json_utils::{JsonFormat, extract_jsonl},
    sparse_dataset::{SparseDatasetMutTrait, SparseDatasetStableTrait, SparseDatasetTrait},
};

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{InvertedIndex, SpaceUsage, inverted_index::Configuration, inverted_index::Knn};

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<S>
where
    S: SparseDatasetTrait,
{
    #[serde(bound(
        serialize = "S: Serialize, S::Component: Serialize",
        deserialize = "S: DeserializeOwned, S::Component: DeserializeOwned"
    ))]
    inverted_index: InvertedIndex<S>,
    document_mapping: Option<Box<[String]>>,
    token_to_id_map: HashMap<String, usize>,
}

impl<S> SpaceUsage for SeismicIndex<S>
where
    S: SparseDatasetTrait,
{
    fn space_usage_byte(&self) -> usize {
        //TODO: add the SpaceUsage of document_mapping and token_to_id_map
        self.inverted_index.space_usage_byte()
    }
}

impl<S> SeismicIndex<S>
where
    S: SparseDatasetStableTrait + Sync,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        self.inverted_index.get_doc_ids_in_postings(list_id)
    }

    pub fn new(
        dataset: S,
        config: Configuration,
        document_mapping: Option<impl Into<Box<[String]>>>,
        token_to_id_map: HashMap<String, usize>,
    ) -> Self {
        let inverted_index = InvertedIndex::build(dataset, config);

        Self {
            inverted_index,
            document_mapping: document_mapping.map(|d| d.into()),
            token_to_id_map,
        }
    }

    pub fn remap_doc_ids(
        &self,
        plain_results: impl IntoIterator<Item = (f32, usize)>,
        query_id: &str,
    ) -> Vec<(String, f32, String)> {
        match &self.document_mapping {
            Some(mapping) => plain_results
                .into_iter()
                .map(|(distance, doc_id)| (query_id.to_owned(), distance, mapping[doc_id].clone()))
                .collect(),
            None => plain_results
                .into_iter()
                .map(|(distance, doc_id)| (query_id.to_owned(), distance, doc_id.to_string()))
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
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
                .filter_map(|(qc, &qv)| {
                    self.token_to_id_map
                        .get(qc)
                        .map(|id| (S::Component::from_usize(*id).unwrap(), qv))
                })
                .sorted_by(|(a, _), (b, _)| a.cmp(b))
                .unzip();

        self.inverted_index.search(
            &filtered_query_components,
            &filtered_query_values,
            k,
            query_cut,
            heap_factor,
            n_knn,
            first_sorted,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        query_id: &str,
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

    pub fn process_data<T>(
        reader: BufReader<impl std::io::Read>,
        row_count: usize,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> (S, Vec<String>, HashMap<String, usize>)
    where
        T: SparseDatasetMutTrait,
        S: FromDatasetGenericF32<T>,
    {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let mut token_to_id_mapping = HashMap::<String, usize>::new();
        let mut converted_data = T::default();

        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values) = extract_jsonl::<T::Value>(x.unwrap());
            doc_id_mapping.push(doc_id);

            let ids: Vec<_> = match &input_token_to_id_map {
                None => {
                    for token in tokens.iter() {
                        if !token_to_id_mapping.contains_key(token) {
                            token_to_id_mapping
                                .insert(token.to_string(), token_to_id_mapping.len());
                        }
                    }
                    let n_bits = size_of::<T::Component>() as u32 * 8;
                    assert!(
                        token_to_id_mapping.len() < 2_usize.pow(n_bits),
                        "The number of different tokens exceeds 2^{}.",
                        n_bits
                    );
                    tokens
                        .into_iter()
                        .map(|t| T::Component::from_usize(token_to_id_mapping[&t]).unwrap())
                        .collect()
                }
                Some(valid_mapping) => tokens
                    .into_iter()
                    .map(|t| T::Component::from_usize(valid_mapping[&t]).unwrap())
                    .collect(),
            };

            let converted_iterator = ids.into_iter().zip(values).sorted_by_key(|&(c, _)| c);

            converted_data.push_iterator(converted_iterator);
        }

        let final_data = S::from_dataset_f32(converted_data);

        (final_data, doc_id_mapping, token_to_id_mapping)
    }

    pub fn from_file<T>(
        file_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Result<Self, io::Error>
    where
        T: SparseDatasetMutTrait,
        S: FromDatasetGenericF32<T>,
    {
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

    pub fn from_json<T>(
        json_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self
    where
        T: SparseDatasetMutTrait,
        S: FromDatasetGenericF32<T>,
    {
        println!("Reading the collection..");
        let start = Instant::now();

        //read the file and count rows
        let f = File::open(json_path).unwrap_or_else(|_| panic!("Unable to open {}", json_path));
        let reader = io::BufReader::new(f);
        let row_count = reader.lines().count();

        println!("Number of rows: {}", row_count);

        let f = File::open(json_path).unwrap_or_else(|_| panic!("Unable to open {}", json_path));
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
    pub fn from_tar<T>(
        tar_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self
    where
        T: SparseDatasetMutTrait,
        S: FromDatasetGenericF32<T>,
    {
        println!("Reading the collection..");
        let start = Instant::now();

        //decompress gz, extract first file (json) of the archive
        //read the file and count rows
        let tar_gz_file =
            File::open(tar_path).unwrap_or_else(|_| panic!("Unable to open {}", tar_path));
        let gz_decoder = GzDecoder::new(tar_gz_file);
        let mut archive = Archive::new(gz_decoder);
        let json_entry = archive.entries().unwrap().next().unwrap().unwrap();
        let reader = BufReader::new(json_entry);
        let row_count = reader.lines().count();

        println!("Number of rows: {}", row_count);

        //Deserialize json
        let tar_gz_file =
            File::open(tar_path).unwrap_or_else(|_| panic!("Unable to open {}", tar_path));
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

    pub fn is_empty(&self) -> bool {
        self.inverted_index.is_empty()
    }

    pub fn inverted_index(&self) -> &InvertedIndex<S> {
        &self.inverted_index
    }

    pub fn dataset(&self) -> &S {
        self.inverted_index.dataset()
    }

    pub fn add_knn(&mut self, knn: Knn) {
        self.inverted_index.add_knn(knn);
    }

    pub fn knn_len(&self) -> usize {
        self.inverted_index.knn_len()
    }

    pub fn from_dataset<T>(dataset: SeismicDataset<T>, config: Configuration) -> Self
    where
        T: SparseDatasetMutTrait,
        S: From<T> + SparseDatasetStableTrait,
    {
        Self {
            inverted_index: InvertedIndex::build(S::from(dataset.sparse_dataset), config),
            document_mapping: Some(dataset.document_mapping.into_boxed_slice()),
            token_to_id_map: dataset.token_to_id_map,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeismicDataset<S>
where
    S: SparseDatasetMutTrait,
{
    sparse_dataset: S,
    document_mapping: Vec<String>,
    token_to_id_map: HashMap<String, usize>,
}

impl<S> Default for SeismicDataset<S>
where
    S: SparseDatasetMutTrait,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> SeismicDataset<S>
where
    S: SparseDatasetMutTrait,
{
    pub fn new() -> Self {
        let sparse_dataset = S::default();
        let document_mapping = Vec::<String>::new();
        let token_to_id_map = HashMap::new();
        Self {
            sparse_dataset,
            document_mapping,
            token_to_id_map,
        }
    }

    pub fn sparse_dataset(self) -> S {
        self.sparse_dataset
    }

    pub fn token_to_id_map(&self) -> &HashMap<String, usize> {
        &self.token_to_id_map
    }

    pub fn len(&self) -> usize {
        self.sparse_dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sparse_dataset.is_empty()
    }

    pub fn add_document(&mut self, id: String, tokens: &[String], values: &[f32]) {
        self.document_mapping.push(id);

        let mut components: Vec<S::Component> = Vec::with_capacity(tokens.len());
        for c in tokens.iter() {
            let next_token_id = self.token_to_id_map.len();
            components.push(
                S::Component::from_usize(
                    *self
                        .token_to_id_map
                        .entry(c.to_string())
                        .or_insert_with(|| next_token_id),
                )
                .unwrap(),
            );
            let n_bits = size_of::<S::Component>() as u32 * 8;
            assert!(
                self.token_to_id_map.len() < 2_usize.pow(n_bits),
                "The number of different tokens exceeds 2^{}.",
                n_bits
            );
        }

        let sorted_components_values = components
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, S::Value::from_f32(values[i]).unwrap()))
            .sorted_by_key(|(c, _)| *c);

        self.sparse_dataset.push_iterator(sorted_components_values);
    }

    pub fn search(
        &self,
        query_id: &str,
        query_components: &[String],
        query_values: &[f32],
        k: usize,
    ) -> Vec<(String, f32, String)> {
        let filtered_query = query_components
            .iter()
            .zip(query_values)
            .filter_map(|(qc, &qv)| self.token_to_id_map.get(qc).map(|id| (*id, qv)))
            .sorted_by(|(a, _), (b, _)| a.cmp(b));

        let plain_results = self.sparse_dataset.search(filtered_query, k);

        self.remap_doc_ids(plain_results, query_id)
    }

    pub fn remap_doc_ids(
        &self,
        plain_results: Vec<(f32, usize)>,
        query_id: &str,
    ) -> Vec<(String, f32, String)> {
        let remapped_results: Vec<(String, f32, String)> = plain_results
            .iter()
            .map(|(distance, doc_id)| {
                (
                    query_id.to_string(),
                    *distance,
                    self.document_mapping[*doc_id].clone(),
                )
            })
            .collect();

        remapped_results
    }
}
