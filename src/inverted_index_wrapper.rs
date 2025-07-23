use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    time::Instant,
};

use crate::{
    json_utils::{extract_jsonl, JsonFormat},
    ComponentType,
};

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{
    inverted_index::Configuration, inverted_index::Knn, InvertedIndex, SpaceUsage, SparseDataset,
    SparseDatasetMut, ValueType,
};

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    inverted_index: InvertedIndex<C, T>,
    document_mapping: Option<Box<[String]>>,
    token_to_id_map: HashMap<String, C>,
}

impl<C, T> SpaceUsage for SeismicIndex<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    fn space_usage_byte(&self) -> usize {
        //TODO: add the SpaceUsage of document_mapping and token_to_id_map
        self.inverted_index.space_usage_byte()
    }
}

impl<C, T> SeismicIndex<C, T>
where
    C: ComponentType,
    T: PartialOrd + ValueType,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        self.inverted_index.get_doc_ids_in_postings(list_id)
    }

    pub fn new(
        dataset: SparseDataset<C, T>,
        config: Configuration,
        document_mapping: Option<Vec<String>>,
        token_to_id_map: HashMap<String, C>,
    ) -> Self
    where
        T: ValueType + PartialOrd,
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
        query_id: &str,
    ) -> Vec<(String, f32, String)> {
        let remapped_results: Vec<(String, f32, String)> = match &self.document_mapping {
            Some(mapp) => plain_results
                .iter()
                .map(|(distance, doc_id)| (query_id.to_owned(), *distance, mapp[*doc_id].clone()))
                .collect(),
            None => plain_results
                .iter()
                .map(|(distance, doc_id)| (query_id.to_owned(), *distance, doc_id.to_string()))
                .collect(),
        };

        remapped_results
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
                .filter(|(qc, _)| self.token_to_id_map.contains_key(*qc))
                .map(|(qc, &qv)| (self.token_to_id_map[qc], qv))
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

    pub fn process_data(
        reader: BufReader<impl std::io::Read>,
        row_count: usize,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> (SparseDataset<C, T>, Vec<String>, HashMap<String, C>) {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let mut token_to_id_mapping = HashMap::<String, C>::new();
        let mut converted_data = SparseDatasetMut::<C, T>::default();

        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values) = extract_jsonl::<T>(x.unwrap());
            doc_id_mapping.push(doc_id);

            let ids: Vec<_> = match &input_token_to_id_map {
                None => {
                    for token in tokens.iter() {
                        if !token_to_id_mapping.contains_key(token) {
                            token_to_id_mapping.insert(
                                token.to_string(),
                                C::from_usize(token_to_id_mapping.len()).unwrap(),
                            );
                        }
                    }
                    // moved outside of the loop to avoid checking every token
                    if C::WIDTH == 16 {
                        assert!(
                            token_to_id_mapping.len() < u16::MAX as usize,
                            "The number of different tokens exceeds 2**16. "
                        );
                    } else if C::WIDTH == 32 {
                        assert!(
                            token_to_id_mapping.len() < u32::MAX as usize,
                            "The number of different tokens exceeds 2**32. "
                        );
                    }
                    tokens.iter().map(|t| token_to_id_mapping[t]).collect()
                }
                Some(valid_mapping) => tokens
                    .iter()
                    .map(|t| C::from_usize(valid_mapping[t]).unwrap())
                    .collect(),
            };

            let converted_vector: Vec<(C, T)> = ids
                .into_iter()
                .zip(values)
                .sorted_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
                .collect();

            converted_data.push_pairs(&converted_vector[..]);
        }

        let final_data: SparseDataset<C, T> = SparseDataset::<C, T>::from(converted_data);

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

    pub fn from_tar(
        tar_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self {
        println!("Reading the collection...");
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

    pub fn inverted_index(&self) -> &InvertedIndex<C, T> {
        &self.inverted_index
    }

    pub fn dataset(&self) -> &SparseDataset<C, T> {
        self.inverted_index.dataset()
    }

    pub fn add_knn(&mut self, knn: Knn) {
        self.inverted_index.add_knn(knn);
    }

    pub fn knn_len(&self) -> usize {
        self.inverted_index.knn_len()
    }

    pub fn from_dataset(dataset: SeismicDataset<C, T>, config: Configuration) -> Self {
        Self {
            inverted_index: InvertedIndex::build(
                SparseDataset::from(dataset.sparse_dataset),
                config,
            ),
            document_mapping: Some(dataset.document_mapping.into_boxed_slice()),
            token_to_id_map: dataset.token_to_id_map.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeismicDataset<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    sparse_dataset: SparseDatasetMut<C, T>,
    document_mapping: Vec<String>,
    token_to_id_map: HashMap<String, C>,
}

impl<C, T> Default for SeismicDataset<C, T>
where
    C: ComponentType,
    T: PartialOrd + ValueType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C, T> SeismicDataset<C, T>
where
    C: ComponentType,
    T: PartialOrd + ValueType,
{
    pub fn new() -> Self
    where
        T: ValueType + PartialOrd,
    {
        let sparse_dataset = SparseDatasetMut::new();
        let document_mapping = Vec::<String>::new();
        let token_to_id_map = HashMap::<String, C>::new();
        Self {
            sparse_dataset,
            document_mapping,
            token_to_id_map,
        }
    }

    pub fn sparse_dataset(self) -> SparseDatasetMut<C, T> {
        self.sparse_dataset
    }

    pub fn token_to_id_map(&self) -> &HashMap<String, C> {
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

        let mut components: Vec<C> = Vec::<_>::with_capacity(tokens.len());
        for c in tokens.iter() {
            let next_token_id = self.token_to_id_map.len();
            components.push(
                *self
                    .token_to_id_map
                    .entry(c.to_string())
                    .or_insert_with(|| C::from_usize(next_token_id).unwrap()),
            );
            // moved outside of the loop to avoid checking every token
            if C::WIDTH == 16 {
                assert!(
                    self.token_to_id_map.len() < u16::MAX as usize,
                    "The number of different tokens exceeds 2**16. "
                );
            } else if C::WIDTH == 32 {
                assert!(
                    self.token_to_id_map.len() < u32::MAX as usize,
                    "The number of different tokens exceeds 2**32. "
                );
            }
        }

        let (sorted_indexes, sorted_components): (Vec<_>, Vec<_>) = components
            .into_iter()
            .enumerate()
            .sorted_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unzip();

        let sorted_values = sorted_indexes
            .iter()
            .map(|&i| T::from_f32(values[i]).unwrap())
            .collect::<Vec<_>>();

        self.sparse_dataset.push(&sorted_components, &sorted_values);
    }
}
