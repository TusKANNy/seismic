use crate::inverted_index::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, Knn, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};

use crate::SeismicIndex as Index;
use half::f16;
use half::slice::HalfFloatSliceExt;

use indicatif::ParallelProgressIterator;
use numpy::{PyArrayMethods, PyFixedUnicode, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::iter::{ParallelBridge, ParallelIterator};

use std::collections::HashMap;
use std::fs;

use crate::{InvertedIndex, SparseDataset};
use rayon::prelude::*;

//TODO: write SEISMIC_STRING as a function of MAX_TOKEN_LEN
const MAX_TOKEN_LEN: usize = 30;
const SEISMIC_STRING: &str = "S30";

const MAX_FRACTION: f32 = 1.5;
const DOC_CUT: usize = 15;

#[pyfunction]
pub fn get_seismic_string() -> &'static str {
    SEISMIC_STRING
}

#[pyclass]
pub struct SeismicIndex {
    index: Index<f16>,
}

#[pymethods]
impl SeismicIndex {
    #[getter]
    pub fn get_dim(&self) -> PyResult<usize> {
        Ok(self.index.dim())
    }

    #[getter]
    pub fn get_len(&self) -> PyResult<usize> {
        Ok(self.index.len())
    }

    #[getter]
    pub fn get_nnz(&self) -> PyResult<usize> {
        Ok(self.index.nnz())
    }

    #[getter]
    pub fn knn_len(&self) -> PyResult<usize> {
        Ok(self.index.knn_len())
    }

    pub fn print_space_usage_byte(&self) {
        self.index.print_space_usage_byte();
    }

    pub fn get(&self, id: usize) -> PyResult<(Vec<u16>, Vec<f32>)> {
        let entry = self.index.dataset().get(id);
        Ok((entry.0.to_vec(), entry.1.to_f32_vec()))
    }

    pub fn vector_len(&self, id: usize) -> PyResult<usize> {
        Ok(self.index.dataset().vector_len(id))
    }

    #[staticmethod]
    pub fn load(index_path: &str) -> PyResult<SeismicIndex> {
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let index = bincode::deserialize::<Index<f16>>(&serialized).unwrap();
        Ok(SeismicIndex { index })
    }

    pub fn save(&self, path: &str) {
        let serialized = bincode::serialize(&self.index).unwrap();
        let path = path.to_string() + ".index.seismic";
        println!("Saving ... {}", path);
        let r = fs::write(path, serialized);
        println!("{:?}", r);
    }

    // Build knn and add it to the inverted index
    pub fn build_knn(&mut self, nknn: usize) {
        let knn = Knn::new(self.index.inverted_index(), nknn);
        self.index.add_knn(knn);
    }

    pub fn save_knn(&self, path: &str) {
        self.index
            .inverted_index()
            .knn()
            .unwrap()
            .serialize(path)
            .unwrap();
    }

    #[pyo3(signature = (knn_path, nknn=None))]
    pub fn load_knn(&mut self, knn_path: &str, nknn: Option<usize>) {
        let knn = Knn::new_from_serialized(knn_path, nknn);
        self.index.add_knn(knn);
    }

    /*
    Order of attributes matters:
    Rust processes attributes sequentially, and attributes like #[pyo3(...)] and #[staticmethod] interact with each other. By placing:
    #[allow(clippy::too_many_arguments)] above #[staticmethod], it ensures Clippy's lint suppression happens first without interfering
    with Pyo3's attribute processing. When #[staticmethod] comes first, it can cause unexpected behavior if the following attributes
    aren't interpreted correctly.                                                                                                   */
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (input_path, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, nknn=0, knn_path=None, batched_indexing=None, input_token_to_id_map=None, num_threads=0))]
    pub fn build(
        input_path: &str,
        n_postings: usize,
        centroid_fraction: f32,
        min_cluster_size: usize,
        summary_energy: f32,
        nknn: usize,
        knn_path: Option<String>,
        batched_indexing: Option<usize>,
        input_token_to_id_map: Option<HashMap<String, usize>>,
        num_threads: usize,
    ) -> PyResult<SeismicIndex> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let knn_config = KnnConfiguration::new(nknn, knn_path);

        let config = Configuration::default()
            .pruning_strategy(PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction: MAX_FRACTION,
            })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction,
                min_cluster_size,
                clustering_algorithm: ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                    doc_cut: DOC_CUT,
                },
            })
            .summarization_strategy(SummarizationStrategy::EnergyPreserving { summary_energy })
            .knn(knn_config)
            .batched_indexing(batched_indexing);

        println!("\nBuilding the index...");
        println!("{:?}", config);

        let index =
            Index::from_file(&input_path.to_owned(), config, input_token_to_id_map).unwrap();

        Ok(SeismicIndex { index })
    }

    //PyFixedUnicode is required to handle non ascii characters in tokens
    #[pyo3(signature = (query_id, query_components, query_values, k, query_cut, heap_factor, n_knn=0, sorted=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn search<'py>(
        &self,
        query_id: String,
        query_components: PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>,
        query_values: PyReadonlyArrayDyn<'py, f32>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool,
    ) -> Vec<(String, f32, String)> {
        self.index.search(
            &query_id,
            &query_components
                .to_vec()
                .unwrap()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>(),
            &query_values.to_vec().unwrap(),
            k,
            query_cut,
            heap_factor,
            n_knn,
            sorted,
        )
    }
    #[pyo3(signature = (queries_ids, query_components, query_values, k, query_cut, heap_factor, n_knn=0, sorted=true, num_threads=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search<'py>(
        &self,
        queries_ids: PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>,
        query_components: Bound<'py, PyList>,
        query_values: Bound<'_, PyList>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool,
        num_threads: usize,
    ) -> Vec<Vec<(String, f32, String)>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let qv: Vec<Vec<f32>> = query_values
            .iter()
            .map(|i| {
                i.extract::<PyReadonlyArrayDyn<f32>>()
                    .unwrap()
                    .to_vec()
                    .unwrap()
            })
            .collect();

        let qc = query_components
            .iter()
            .map(|i| {
                let array = i
                    .extract::<PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>>()
                    .unwrap();
                array
                    .to_vec()
                    .unwrap()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let results: Vec<_> = queries_ids
            .to_vec()
            .unwrap()
            .iter()
            .zip(qc.iter())
            .zip(qv.iter())
            .par_bridge()
            .progress_count(queries_ids.len().unwrap() as u64)
            .map(|((query_id, components), values)| {
                self.index.search(
                    &query_id.to_string(),
                    components,
                    values,
                    k,
                    query_cut,
                    heap_factor,
                    n_knn,
                    sorted,
                )
            })
            .collect();

        results
    }
}

#[pyclass]
pub struct SeismicIndexRaw {
    inverted_index: InvertedIndex<f16>,
}

#[pymethods]
impl SeismicIndexRaw {
    #[getter]
    pub fn get_dim(&self) -> PyResult<usize> {
        Ok(self.inverted_index.dim())
    }

    #[getter]
    pub fn get_len(&self) -> PyResult<usize> {
        Ok(self.inverted_index.len())
    }

    #[getter]
    pub fn get_nnz(&self) -> PyResult<usize> {
        Ok(self.inverted_index.nnz())
    }

    #[getter]
    pub fn knn_len(&self) -> PyResult<usize> {
        Ok(self.inverted_index.knn_len())
    }

    #[getter]
    pub fn get_is_empty(&self) -> PyResult<bool> {
        Ok(self.inverted_index.is_empty())
    }

    pub fn print_space_usage_byte(&self) {
        self.inverted_index.print_space_usage_byte();
    }

    pub fn get(&self, id: usize) -> PyResult<(Vec<u16>, Vec<f32>)> {
        let entry = self.inverted_index.dataset().get(id);
        Ok((entry.0.to_vec(), entry.1.to_f32_vec()))
    }

    pub fn vector_len(&self, id: usize) -> PyResult<usize> {
        Ok(self.inverted_index.dataset().vector_len(id))
    }

    #[staticmethod]
    pub fn load(index_path: &str) -> PyResult<SeismicIndexRaw> {
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let inverted_index = bincode::deserialize::<InvertedIndex<f16>>(&serialized).unwrap();
        Ok(SeismicIndexRaw { inverted_index })
    }

    pub fn save(&self, path: &str) {
        let serialized = bincode::serialize(&self.inverted_index).unwrap();
        let path = path.to_string() + ".index.seismic";
        println!("Saving ... {}", path);
        let r = fs::write(path, serialized);
        println!("{:?}", r);
    }

    // Build knn and add it to the inverted index
    pub fn build_knn(&mut self, nknn: usize) {
        let knn = Knn::new(&self.inverted_index, nknn);
        self.inverted_index.add_knn(knn);
    }

    pub fn save_knn(&self, path: &str) {
        self.inverted_index.knn().unwrap().serialize(path).unwrap();
    }

    #[pyo3(signature = (knn_path, nknn=None))]
    pub fn load_knn(&mut self, knn_path: &str, nknn: Option<usize>) {
        let knn = Knn::new_from_serialized(knn_path, nknn);
        self.inverted_index.add_knn(knn);
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_file, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, nknn=0, knn_path=None, batched_indexing=None))]
    pub fn build(
        input_file: &str,
        n_postings: usize,
        centroid_fraction: f32,
        min_cluster_size: usize,
        summary_energy: f32,
        nknn: usize,
        knn_path: Option<String>,
        batched_indexing: Option<usize>,
    ) -> PyResult<SeismicIndexRaw> {
        let dataset = SparseDataset::<f32>::read_bin_file(input_file)
            .unwrap()
            .quantize_f16();

        let knn_config = KnnConfiguration::new(nknn, knn_path);

        let config = Configuration::default()
            .pruning_strategy(PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction: 1.5,
            })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction,
                min_cluster_size,
                clustering_algorithm: ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                    doc_cut: 15,
                },
            })
            .summarization_strategy(SummarizationStrategy::EnergyPreserving { summary_energy })
            .knn(knn_config)
            .batched_indexing(batched_indexing);
        println!("\nBuilding the index...");
        println!("{:?}", config);

        let inverted_index = InvertedIndex::build(dataset, config);
        Ok(SeismicIndexRaw { inverted_index })
    }

    pub fn search<'py>(
        &self,
        query_components: PyReadonlyArrayDyn<'py, i32>,
        query_values: PyReadonlyArrayDyn<'py, f32>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool,
    ) -> Vec<(f32, usize)> {
        self.inverted_index.search(
            &query_components
                .to_vec()
                .unwrap()
                .iter()
                .map(|x| *x as u16)
                .collect::<Vec<_>>(),
            &query_values.to_vec().unwrap(),
            k,
            query_cut,
            heap_factor,
            n_knn,
            sorted, // first_sorted is set to false
        )
    }

    #[pyo3(signature = (query_path, k, query_cut, heap_factor, n_knn, sorted, num_threads=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search<'py>(
        &self,
        query_path: &str,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool,
        num_threads: usize,
    ) -> Vec<Vec<(f32, usize)>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let queries = SparseDataset::<f32>::read_bin_file(query_path).unwrap();

        queries
            .par_iter()
            .map(|query| {
                self.inverted_index.search(
                    query.0,
                    query.1,
                    k,
                    query_cut,
                    heap_factor,
                    n_knn,
                    sorted,
                )
            })
            .collect::<Vec<_>>()
    }
}
