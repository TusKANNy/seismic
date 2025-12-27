use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    time::Instant,
};

use crate::json_utils::{JsonFormat, extract_jsonl};
use crate::{ComponentType, ValueType};
use half::f16;
use vectorium::{
    Dataset, Distance, DotProduct, GrowableDataset, SparseDataset, SparseDatasetGrowable,
    SparseQuantizer,
    SparseVector1D, SpaceUsage, Vector1D, VectorEncoder,
};
use vectorium::dataset::ScoredVectorDotProduct;

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{InvertedIndex, inverted_index::Configuration, inverted_index::Knn};

type ComponentFor<E> = <E as VectorEncoder>::OutputComponentType;
type ValueFor<E> = <E as VectorEncoder>::OutputValueType;
type QueryValueFor<E> = <E as VectorEncoder>::QueryValueType;
type SparseEncodedVector<'a, E> = SparseVector1D<
    ComponentFor<E>,
    ValueFor<E>,
    &'a [ComponentFor<E>],
    &'a [ValueFor<E>],
>;
type ScoredVectorFor = ScoredVectorDotProduct;

#[derive(Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<S, E>
where
    S: Dataset<E>,
    E: VectorEncoder,
    ComponentFor<E>: ComponentType,
{
    #[serde(bound(
        serialize = "S: Serialize, ComponentFor<E>: Serialize",
        deserialize = "S: DeserializeOwned, ComponentFor<E>: DeserializeOwned"
    ))]
    inverted_index: InvertedIndex<S, E>,
    document_mapping: Option<Box<[String]>>,
    token_to_id_map: HashMap<String, usize>,
}

impl<S, E> SpaceUsage for SeismicIndex<S, E>
where
    S: Dataset<E> + SpaceUsage,
    E: VectorEncoder,
    ComponentFor<E>: ComponentType,
{
    fn space_usage_byte(&self) -> usize {
        //TODO: add the SpaceUsage of document_mapping and token_to_id_map
        self.inverted_index.space_usage_byte()
    }
}

impl<S, E> SeismicIndex<S, E>
where
    S: Dataset<E> + Sync + SpaceUsage,
    E: VectorEncoder<QueryValueType = f32, Distance = DotProduct>,
    E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
    E: SparseQuantizer<InputComponentType = ComponentFor<E>, InputValueType = f32>,
    E: vectorium::SpaceUsage,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
    S: From<SparseDataset<E>>,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    QueryValueFor<E>: ValueType,
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
        plain_results: impl IntoIterator<Item = ScoredVectorFor>,
        query_id: &str,
    ) -> Vec<(String, f32, String)> {
        match &self.document_mapping {
            Some(mapping) => plain_results
                .into_iter()
                .map(|result| {
                    let doc_id = result.vector as usize;
                    (
                        query_id.to_owned(),
                        result.distance.distance(),
                        mapping[doc_id].clone(),
                    )
                })
                .collect(),
            None => plain_results
                .into_iter()
                .map(|result| {
                    let doc_id = result.vector as usize;
                    (query_id.to_owned(), result.distance.distance(), doc_id.to_string())
                })
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
    ) -> Vec<ScoredVectorFor> {
        let (filtered_query_components, filtered_query_values): (Vec<_>, Vec<_>) =
            query_components_original
                .iter()
                .zip(query_values)
                .filter_map(|(qc, &qv)| {
                    self.token_to_id_map
                        .get(qc)
                        .map(|id| (ComponentFor::<E>::from_usize(*id).unwrap(), qv))
                })
                .sorted_by(|(a, _), (b, _)| a.cmp(b))
                .unzip();

        let query = SparseVector1D::new(filtered_query_components, filtered_query_values);
        self.inverted_index.search(
            &query,
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
        token_to_id_mapping: HashMap<String, usize>,
    ) -> (S, Vec<String>, HashMap<String, usize>)
    where
        E: SparseQuantizer<InputComponentType = ComponentFor<E>, InputValueType = f32>,
        for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
        ComponentFor<E>: ComponentType,
        S: From<SparseDataset<E>>,
    {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let mut converted_data = SparseDatasetGrowable::<E>::new(E::new(
            token_to_id_mapping.len(),
            token_to_id_mapping.len(),
        ));

        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values) = extract_jsonl::<f32>(x.unwrap());
            doc_id_mapping.push(doc_id);

            let ids: Vec<_> = tokens
                .into_iter()
                .map(|t| ComponentFor::<E>::from_usize(token_to_id_mapping[&t]).unwrap())
                .collect();

            let converted_iterator = ids.into_iter().zip(values).sorted_by_key(|&(c, _)| c);
            let (components, values): (Vec<_>, Vec<_>) = converted_iterator.unzip();

            converted_data.push(SparseVector1D::new(components, values));
        }

        let frozen: SparseDataset<E> = converted_data.into();
        let final_data: S = frozen.into();

        (final_data, doc_id_mapping, token_to_id_mapping)
    }

    fn build_token_map(
        reader: BufReader<impl std::io::Read>,
        row_count: usize,
    ) -> HashMap<String, usize> {
        let mut token_to_id_mapping = HashMap::<String, usize>::new();
        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (_doc_id, tokens, _values) = extract_jsonl::<f32>(x.unwrap());
            for token in tokens {
                if !token_to_id_mapping.contains_key(&token) {
                    token_to_id_mapping.insert(token, token_to_id_mapping.len());
                }
            }
        }

        let n_bits = size_of::<ComponentFor<E>>() as u32 * 8;
        assert!(
            token_to_id_mapping.len() < 2_usize.pow(n_bits),
            "The number of different tokens exceeds 2^{}.",
            n_bits
        );

        token_to_id_mapping
    }

    pub fn from_file(
        file_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Result<Self, io::Error>
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

    pub fn from_json(
        json_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Self
    {
        println!("Reading the collection..");
        let start = Instant::now();

        //read the file and count rows
        let f = File::open(json_path).unwrap_or_else(|_| panic!("Unable to open {}", json_path));
        let reader = io::BufReader::new(f);
        let row_count = reader.lines().count();

        println!("Number of rows: {}", row_count);

        let token_to_id_mapping = match input_token_to_id_map {
            Some(mapping) => mapping,
            None => {
                let f = File::open(json_path)
                    .unwrap_or_else(|_| panic!("Unable to open {}", json_path));
                let reader = BufReader::new(f);
                Self::build_token_map(reader, row_count)
            }
        };

        let f = File::open(json_path).unwrap_or_else(|_| panic!("Unable to open {}", json_path));
        let reader = BufReader::new(f);

        let (final_data, doc_id_mapping, token_to_id_mapping) =
            Self::process_data(reader, row_count, token_to_id_mapping);

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
    ) -> Self
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

        let token_to_id_mapping = match input_token_to_id_map {
            Some(mapping) => mapping,
            None => {
                let tar_gz_file =
                    File::open(tar_path).unwrap_or_else(|_| panic!("Unable to open {}", tar_path));
                let gz_decoder = GzDecoder::new(tar_gz_file);
                let mut archive = Archive::new(gz_decoder);
                let json_entry = archive.entries().unwrap().next().unwrap().unwrap();
                let reader = BufReader::new(json_entry);
                Self::build_token_map(reader, row_count)
            }
        };

        //Deserialize json
        let tar_gz_file =
            File::open(tar_path).unwrap_or_else(|_| panic!("Unable to open {}", tar_path));
        let gz_decoder = GzDecoder::new(tar_gz_file);
        let mut archive = Archive::new(gz_decoder);
        let json_entry = archive.entries().unwrap().next().unwrap().unwrap();
        let reader = BufReader::new(json_entry);

        let (final_data, doc_id_mapping, token_to_id_mapping) =
            Self::process_data(reader, row_count, token_to_id_mapping);

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

    pub fn inverted_index(&self) -> &InvertedIndex<S, E> {
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

    pub fn from_dataset(dataset: SeismicDataset<ComponentFor<E>>, config: Configuration) -> Self
    where
        S: From<SparseDataset<E>>,
        E: SparseQuantizer<InputComponentType = ComponentFor<E>, InputValueType = f32>,
        for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
        ComponentFor<E>: ComponentType,
        ValueFor<E>: ValueType,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
    {
        let dim = dataset.sparse_dataset.input_dim();
        let quantizer = E::new(dim, dim);
        let mut converted = SparseDatasetGrowable::<E>::new(quantizer);
        for vec in dataset.sparse_dataset.iter() {
            let components = vec.components_as_slice().to_vec();
            let values: Vec<_> = vec.values_as_slice().iter().map(|v| v.to_f32().unwrap()).collect();
            converted.push(SparseVector1D::new(components, values));
        }

        let frozen: SparseDataset<E> = converted.into();
        Self {
            inverted_index: InvertedIndex::build(frozen.into(), config),
            document_mapping: Some(dataset.document_mapping.into_boxed_slice()),
            token_to_id_map: dataset.token_to_id_map,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeismicDataset<C>
where
    C: ComponentType,
{
    sparse_dataset: vectorium::PlainSparseDatasetGrowable<C, f16, vectorium::DotProduct>,
    document_mapping: Vec<String>,
    token_to_id_map: HashMap<String, usize>,
}

impl<C> Default for SeismicDataset<C>
where
    C: ComponentType + std::convert::TryFrom<usize>,
    <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> SeismicDataset<C>
where
    C: ComponentType + std::convert::TryFrom<usize>,
    <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
{
    pub fn new() -> Self {
        let quantizer =
            vectorium::PlainSparseQuantizer::<C, f16, vectorium::DotProduct>::new(0, 0);
        let sparse_dataset =
            vectorium::PlainSparseDatasetGrowable::<C, f16, vectorium::DotProduct>::new(quantizer);
        let document_mapping = Vec::<String>::new();
        let token_to_id_map = HashMap::new();
        Self {
            sparse_dataset,
            document_mapping,
            token_to_id_map,
        }
    }

    pub fn sparse_dataset(self) -> vectorium::PlainSparseDatasetGrowable<C, f16, vectorium::DotProduct> {
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

    fn ensure_dim(&mut self, dim: usize) {
        if dim <= self.sparse_dataset.input_dim() {
            return;
        }

        let quantizer =
            vectorium::PlainSparseQuantizer::<C, f16, vectorium::DotProduct>::new(dim, dim);
        let mut rebuilt =
            vectorium::PlainSparseDatasetGrowable::<C, f16, vectorium::DotProduct>::new(quantizer);

        for vec in self.sparse_dataset.iter() {
            rebuilt.push(SparseVector1D::new(
                vec.components_as_slice().to_vec(),
                vec.values_as_slice().to_vec(),
            ));
        }

        self.sparse_dataset = rebuilt;
    }

    pub fn add_document(&mut self, id: String, tokens: &[String], values: &[f32]) {
        self.document_mapping.push(id);

        let mut components: Vec<C> = Vec::with_capacity(tokens.len());
        for c in tokens.iter() {
            let next_token_id = self.token_to_id_map.len();
            let token_id = *self
                .token_to_id_map
                .entry(c.to_string())
                .or_insert_with(|| next_token_id);
            components.push(C::try_from(token_id).unwrap());
            let n_bits = size_of::<C>() as u32 * 8;
            assert!(
                self.token_to_id_map.len() < 2_usize.pow(n_bits),
                "The number of different tokens exceeds 2^{}.",
                n_bits
            );
        }

        self.ensure_dim(self.token_to_id_map.len());

        let values: Vec<_> = values.iter().map(|v| f16::from_f32(*v)).collect();
        let sorted_components_values = components
            .into_iter()
            .zip(values)
            .sorted_by_key(|(c, _)| *c);
        let (components, values): (Vec<_>, Vec<_>) = sorted_components_values.unzip();

        self.sparse_dataset
            .push(SparseVector1D::new(components, values));
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
            .sorted_by(|(a, _), (b, _)| a.cmp(b))
            .map(|(id, value)| (C::try_from(id).unwrap(), value));
        let (components, values): (Vec<_>, Vec<_>) = filtered_query.unzip();
        let query = SparseVector1D::new(components, values);
        let plain_results = self
            .sparse_dataset
            .search(query, k)
            .into_iter()
            .map(|result| (result.distance.distance(), result.vector as usize))
            .collect();

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
