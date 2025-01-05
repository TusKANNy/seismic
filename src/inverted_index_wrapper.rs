use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    time::Instant,
};

use crate::{
    ComponentType,
    json_utils::{JsonFormat, extract_jsonl},
};

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{
    InvertedIndex, SpaceUsage, SparseDataset, SparseDatasetMut, ValueType,
    inverted_index::Configuration, inverted_index::Knn,
};

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    inverted_index: InvertedIndex<C, V>,
    document_mapping: Option<Box<[String]>>,
    token_to_id_map: HashMap<String, C>,
}

impl<C, V> SpaceUsage for SeismicIndex<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn space_usage_byte(&self) -> usize {
        //TODO: add the SpaceUsage of document_mapping and token_to_id_map
        self.inverted_index.space_usage_byte()
    }
}

impl<C, V> SeismicIndex<C, V>
where
    C: ComponentType,
    V: PartialOrd + ValueType,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        self.inverted_index.get_doc_ids_in_postings(list_id)
    }

    pub fn new(
        dataset: SparseDataset<C, V>,
        config: Configuration,
        document_mapping: Option<impl Into<Box<[String]>>>,
        token_to_id_map: HashMap<String, C>,
    ) -> Self
    where
        V: ValueType + PartialOrd,
    {
        let inverted_index = InvertedIndex::build(dataset, config);

        Self {
            inverted_index,
            document_mapping: document_mapping.map(|d| d.into()),
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
                .map(|(distance, doc_id)| (query_id.to_string(), *distance, mapp[*doc_id].clone()))
                .collect(),
            None => plain_results
                .iter()
                .map(|(distance, doc_id)| (query_id.to_string(), *distance, doc_id.to_string()))
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
    ) -> (SparseDataset<C, V>, Vec<String>, HashMap<String, C>) {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let mut token_to_id_mapping = HashMap::<String, C>::new();
        let mut converted_data = SparseDatasetMut::<C, V>::default();

        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values) = extract_jsonl::<V>(x.unwrap());
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
                    assert!(
                        token_to_id_mapping.len() < 2_usize.pow(C::WIDTH as u32),
                        "The number of different tokens exceeds 2^{}.",
                        C::WIDTH
                    );
                    tokens.iter().map(|t| token_to_id_mapping[t]).collect()
                }
                Some(valid_mapping) => tokens
                    .iter()
                    .map(|t| C::from_usize(valid_mapping[t]).unwrap())
                    .collect(),
            };

            let converted_iterator = ids.into_iter().zip(values).sorted_by_key(|&(c, _)| c);

            converted_data.push_iterator(converted_iterator);
        }

        let final_data: SparseDataset<C, V> = SparseDataset::<C, V>::from(converted_data);

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

    pub fn inverted_index(&self) -> &InvertedIndex<C, V> {
        &self.inverted_index
    }

    pub fn dataset(&self) -> &SparseDataset<C, V> {
        self.inverted_index.dataset()
    }

    pub fn add_knn(&mut self, knn: Knn) {
        self.inverted_index.add_knn(knn);
    }

    pub fn knn_len(&self) -> usize {
        self.inverted_index.knn_len()
    }

    pub fn from_dataset(dataset: SeismicDataset<C, V>, config: Configuration) -> Self {
        Self {
            inverted_index: InvertedIndex::build(
                SparseDataset::from(dataset.sparse_dataset),
                config,
            ),
            document_mapping: Some(dataset.document_mapping.into_boxed_slice()),
            token_to_id_map: dataset.token_to_id_map,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeismicDataset<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    sparse_dataset: SparseDatasetMut<C, V>,
    document_mapping: Vec<String>,
    token_to_id_map: HashMap<String, C>,
}

impl<C, V> Default for SeismicDataset<C, V>
where
    C: ComponentType,
    V: PartialOrd + ValueType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C, V> SeismicDataset<C, V>
where
    C: ComponentType,
    V: PartialOrd + ValueType,
{
    pub fn new() -> Self
    where
        V: ValueType + PartialOrd,
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

    pub fn sparse_dataset(self) -> SparseDatasetMut<C, V> {
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

    pub fn add_document(&mut self, id: String, tokens: &[String], values: &[f32])
    where
        V: ValueType,
    {
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
            assert!(
                self.token_to_id_map.len() < 2_usize.pow(C::WIDTH as u32),
                "The number of different tokens exceeds 2^{}.",
                C::WIDTH
            );
        }

        let (sorted_indexes, sorted_components): (Vec<_>, Vec<_>) = components
            .into_iter()
            .enumerate()
            .sorted_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unzip();

        let sorted_values = sorted_indexes
            .iter()
            .map(|&i| V::from_f32_saturating(values[i]))
            .collect::<Vec<_>>();

        self.sparse_dataset.push(&sorted_components, &sorted_values);
    }
}
