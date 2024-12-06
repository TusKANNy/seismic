use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Write},
    time::Instant,
};

use crate::json_utils::{extract_jsonl, JsonFormat};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use crate::{
    inverted_index::Configuration, DataType, InvertedIndex, SpaceUsage, SparseDataset,
    SparseDatasetMut, inverted_index::Knn,
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
        let query_components: Vec<u16> = query_components_original
            .iter()
            .map(|qc| self.token_to_id_map[qc] as u16)
            .collect();

        let results = self.inverted_index.search(
            &query_components,
            query_values,
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

    pub fn from_json(
        json_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self {
        println!("Reading the collection..");
        let f = File::open(json_path).expect(&format!("Unable to open {}", json_path));
        let reader = BufReader::new(f);
        let stream: serde_json::StreamDeserializer<
            serde_json::de::IoRead<BufReader<File>>,
            JsonFormat,
        > = Deserializer::from_reader(reader).into_iter();

        let start = Instant::now();

        let mut doc_id_mapping = Vec::with_capacity(1_000_000);

        let mut token_to_id_mapping = HashMap::<String, usize>::with_capacity(30000);

        let mut converted_data = SparseDatasetMut::<T>::default();

        let mut analog_counter = 0;
 
        for x in stream.into_iter() {
            let (doc_id, tokens, values) = extract_jsonl::<T>(x.unwrap());
            doc_id_mapping.push(doc_id);

            let ids = match input_token_to_id_map {
                None => {
                    for token in tokens.iter() {
                        if !token_to_id_mapping.contains_key(token) {
                            token_to_id_mapping
                                .insert(token.to_string(), token_to_id_mapping.len());
                        }
                    }
                    let ids: Vec<u16> = tokens
                        .iter()
                        .map(|t| token_to_id_mapping[t] as u16)
                        .collect();
                    ids
                }

                Some(ref valid_mapping) => {
                    let ids: Vec<u16> = tokens.iter().map(|t| valid_mapping[t] as u16).collect();
                    ids
                }
            };

            let converted_vector: Vec<(u16, T)> = ids
                .into_iter()
                .zip(values)
                .sorted_by(|(a, _), (b, _)| a.partial_cmp(&b).unwrap())
                .collect();

            converted_data.push_pairs(&converted_vector[..]);

            if analog_counter % 10_000 == 0 {
                print!("\rReading record {}", analog_counter);
                let _ = std::io::stdout().flush();
            }
            analog_counter += 1;
        }

        println!();
        println!(
            "Elapsed time to read the collection {:}",
            start.elapsed().as_micros()
        );
        let final_data: SparseDataset<T> = SparseDataset::<T>::from(converted_data);

        Self::new(
            final_data,
            config,
            Some(doc_id_mapping),
            token_to_id_mapping,
        )
    }

    //TODO: implement
    #[allow(unused_variables)]
    pub fn from_tar(
        json_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self {
        todo!()
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

}