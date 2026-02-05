use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    time::Instant,
};

use crate::json_utils::{JsonFormat, extract_jsonl};
use half::f16;
use num_traits::{FromPrimitive, ToPrimitive};
use vectorium::ComponentType;
use vectorium::dataset::ConvertFrom;
use vectorium::dataset::ScoredVector;
use vectorium::vector_encoder::{SparseDataEncoder, SparseVectorEncoder};
use vectorium::{
    Dataset, DatasetGrowable, Distance, DotProduct, QueryEvaluator, ScalarSparseQuantizer,
    SpaceUsage, SparseData, SparseDataset, SparseDatasetGrowable, SparseVectorView, VectorEncoder,
};

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{
    InvertedIndexBase,
    configurations::Configuration,
    index_traits::{IndexBuildDataset, IndexSearchDataset},
    inverted_index::Knn,
};

type EncoderFor<S> = <S as Dataset>::Encoder;
type ComponentFor<S> = <EncoderFor<S> as SparseDataEncoder>::OutputComponentType;
type ValueFor<S> = <EncoderFor<S> as SparseDataEncoder>::OutputValueType;
type ScoredVectorDotProduct = ScoredVector<DotProduct>;

#[derive(Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
    #[serde(bound(
        serialize = "S: Serialize, ComponentFor<S>: Serialize",
        deserialize = "S: DeserializeOwned, ComponentFor<S>: DeserializeOwned"
    ))]
    inverted_index: InvertedIndexBase<S>,
    document_mapping: Option<Box<[String]>>,
    token_to_id_map: HashMap<String, usize>,
}

impl<S> SpaceUsage for SeismicIndex<S>
where
    S: Dataset + SparseData + SpaceUsage,
    EncoderFor<S>: SparseDataEncoder,
    ComponentFor<S>: SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        //TODO: add the SpaceUsage of document_mapping and token_to_id_map
        self.inverted_index.space_usage_bytes()
    }
}

impl<S> SeismicIndex<S>
where
    S: Dataset + SparseData + Sync + IndexBuildDataset,
    EncoderFor<S>: SparseVectorEncoder<InputValueType = f32>,
    S: From<SparseDataset<EncoderFor<S>>>,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        self.inverted_index.get_doc_ids_in_postings(list_id)
    }

    pub fn new(
        dataset: S,
        config: Configuration,
        document_mapping: Option<impl Into<Box<[String]>>>,
        token_to_id_map: HashMap<String, usize>,
    ) -> Self
    where
        ValueFor<S>: vectorium::FromF32,
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
    {
        let inverted_index = InvertedIndexBase::build(dataset, config);

        Self {
            inverted_index,
            document_mapping: document_mapping.map(|d| d.into()),
            token_to_id_map,
        }
    }

    pub fn remap_doc_ids(
        &self,
        plain_results: impl IntoIterator<Item = ScoredVectorDotProduct>,
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
                    (
                        query_id.to_owned(),
                        result.distance.distance(),
                        doc_id.to_string(),
                    )
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
    ) -> Vec<ScoredVectorDotProduct>
    where
        S: IndexSearchDataset,
        ComponentFor<S>: FromPrimitive,
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
    {
        let (filtered_query_components, filtered_query_values): (Vec<_>, Vec<_>) =
            query_components_original
                .iter()
                .zip(query_values)
                .filter_map(|(qc, &qv)| {
                    self.token_to_id_map
                        .get(qc)
                        .and_then(|id| ComponentFor::<S>::from_usize(*id))
                        .map(|component| (component, qv))
                })
                .sorted_by_key(|(component, _)| *component)
                .unzip();

        let query = SparseVectorView::new(
            filtered_query_components.as_slice(),
            filtered_query_values.as_slice(),
        );
        self.inverted_index
            .search(query, k, query_cut, heap_factor, n_knn, first_sorted)
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
    ) -> Vec<(String, f32, String)>
    where
        S: IndexSearchDataset,
        ComponentFor<S>: FromPrimitive,
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
    {
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

    pub fn print_space_usage_byte(&self)
    where
        S: SpaceUsage,
        ComponentFor<S>: SpaceUsage,
    {
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

    pub fn inverted_index(&self) -> &InvertedIndexBase<S> {
        &self.inverted_index
    }

    pub fn dataset(&self) -> &S {
        self.inverted_index.dataset()
    }

    pub fn convert_dataset_into<T>(self) -> SeismicIndex<T>
    where
        T: Dataset + SparseData + ConvertFrom<S> + SpaceUsage,
        EncoderFor<T>: SparseDataEncoder<OutputComponentType = ComponentFor<S>>,
        ComponentFor<T>: SpaceUsage,
    {
        SeismicIndex {
            inverted_index: self.inverted_index.convert_dataset_into(),
            document_mapping: self.document_mapping,
            token_to_id_map: self.token_to_id_map,
        }
    }

    pub fn add_knn(&mut self, knn: Knn) {
        self.inverted_index.add_knn(knn);
    }

    pub fn knn_len(&self) -> usize {
        self.inverted_index.knn_len()
    }
}

type ScalarSparseDataset<C, V> = SparseDataset<ScalarSparseQuantizer<C, f32, V, DotProduct>>;

impl<C, V> SeismicIndex<ScalarSparseDataset<C, V>>
where
    C: ComponentType + FromPrimitive + std::hash::Hash,
    V: vectorium::ValueType + vectorium::Float + vectorium::FromF32,
{
    pub fn from_dataset(dataset: SeismicDataset<C>, config: Configuration) -> Self
    where
        for<'a> <ScalarSparseQuantizer<C, f32, V, DotProduct> as VectorEncoder>::InputVector<'a>:
            From<SparseVectorView<'a, C, f32>>,
    {
        let dim = dataset.sparse_dataset.input_dim();
        let encoder = ScalarSparseQuantizer::<C, f32, V, DotProduct>::new(dim, dim);
        let mut converted =
            SparseDatasetGrowable::<ScalarSparseQuantizer<C, f32, V, DotProduct>>::new(encoder);

        for vec in dataset.sparse_dataset.iter() {
            let components = vec.components().to_vec();
            let values_f32: Vec<f32> = vec.values().iter().map(|v| v.to_f32().unwrap()).collect();
                let input_vec: <ScalarSparseQuantizer<C, f32, V, DotProduct> as VectorEncoder>::InputVector<'_> =
                    SparseVectorView::new(components.as_slice(), values_f32.as_slice());
            converted.push(input_vec);
        }

        let frozen: ScalarSparseDataset<C, V> = converted.into();
        Self::new(
            frozen,
            config,
            Some(dataset.document_mapping),
            dataset.token_to_id_map,
        )
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

        let n_bits = size_of::<C>() as u32 * 8;
        assert!(
            token_to_id_mapping.len() < 2_usize.pow(n_bits),
            "The number of different tokens exceeds 2^{}.",
            n_bits
        );

        token_to_id_mapping
    }

    fn process_data(
        reader: BufReader<impl std::io::Read>,
        row_count: usize,
        token_to_id_mapping: HashMap<String, usize>,
    ) -> (
        ScalarSparseDataset<C, V>,
        Vec<String>,
        HashMap<String, usize>,
    ) {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let dim = token_to_id_mapping.len();
        let encoder = ScalarSparseQuantizer::<C, f32, V, DotProduct>::new(dim, dim);
        let mut converted_data =
            SparseDatasetGrowable::<ScalarSparseQuantizer<C, f32, V, DotProduct>>::new(encoder);

        let stream: serde_json::StreamDeserializer<_, JsonFormat> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values) = extract_jsonl::<f32>(x.unwrap());
            doc_id_mapping.push(doc_id);

            let ids: Vec<_> = tokens
                .into_iter()
                .map(|t| {
                    C::from_usize(token_to_id_mapping[&t])
                        .expect("Failed to convert token id to component type")
                })
                .collect();

            let converted_iterator = ids.into_iter().zip(values).sorted_by_key(|&(c, _)| c);
            let (components, values): (Vec<_>, Vec<_>) = converted_iterator.unzip();
            converted_data.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        let frozen: ScalarSparseDataset<C, V> = converted_data.into();
        (frozen, doc_id_mapping, token_to_id_mapping)
    }

    pub fn from_file(
        file_path: &String,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
    ) -> Result<Self, io::Error> {
        if file_path.ends_with(".jsonl") {
            Ok(Self::from_json(file_path, config, input_token_to_id_map))
        } else if file_path.ends_with(".tar.gz") {
            Ok(Self::from_tar(file_path, config, input_token_to_id_map))
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
        println!("Reading the collection..");
        let start = Instant::now();

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
    ) -> Self {
        println!("Reading the collection..");
        let start = Instant::now();

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
        let sparse_dataset = {
            let quantizer =
                vectorium::PlainSparseQuantizer::<C, f16, vectorium::DotProduct>::new(0, 0);
            vectorium::PlainSparseDatasetGrowable::<C, f16, vectorium::DotProduct>::new(quantizer)
        };
        let document_mapping = Vec::<String>::new();
        let token_to_id_map = HashMap::new();
        Self {
            sparse_dataset,
            document_mapping,
            token_to_id_map,
        }
    }

    pub fn sparse_dataset(
        self,
    ) -> vectorium::PlainSparseDatasetGrowable<C, f16, vectorium::DotProduct> {
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

        let mut rebuilt = {
            let quantizer =
                vectorium::PlainSparseQuantizer::<C, f16, vectorium::DotProduct>::new(dim, dim);
            vectorium::PlainSparseDatasetGrowable::<C, f16, vectorium::DotProduct>::new(quantizer)
        };

        for vec in self.sparse_dataset.iter() {
            let components = vec.components().to_vec();
            let values = vec.values().to_vec();
            rebuilt.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
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

        self.sparse_dataset.push(SparseVectorView::new(
            components.as_slice(),
            values.as_slice(),
        ));
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
        let query = SparseVectorView::new(components.as_slice(), values.as_slice());
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
