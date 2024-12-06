use crate::inverted_index::{
    BlockingStrategy, Configuration, PruningStrategy, SummarizationStrategy, Knn, KnnConfiguration, ClusteringAlgorithm, ClusteringAlgorithmClap
};
use crate::json_utils::read_queries;
use crate::SeismicIndex;
use half::f16;
use half::slice::HalfFloatSliceExt;
use numpy::PyArrayMethods;
use numpy::{PyFixedString, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::fs;

//TODO: write SEISMIC_STRING as a function of MAX_TOKEN_LEN
const MAX_TOKEN_LEN: usize = 30;
const SEISMIC_STRING: &str = "S30";

#[pyfunction]
pub fn get_seismic_string() -> &'static str {
    SEISMIC_STRING
}

#[pyclass]
pub struct PySeismicIndex {
    index: SeismicIndex<f16>,
}

#[pymethods]
impl PySeismicIndex {

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

    pub fn print_space_usage_byte(&self) {
        self.index.print_space_usage_byte();
    }

    /*
        Python does not have built-in support for slices, it does not natively understand the f16
        type because it is not a standard floating-point type (Python primarily uses float which
        corresponds to f64 in Rust). We need to convert f16 values to f32 and then return a
        compatible Python type. https://pyo3.rs/main/conversions/tables.html
    */    
    pub fn get(&self, id: usize) -> PyResult<(Vec<u16>, Vec<f32>)>  {
        let entry = self.index.dataset().get(id);
        Ok((entry.0.to_vec(), entry.1.to_f32_vec()))
    }

    pub fn vector_len(&self, id: usize) -> PyResult<usize> {
        Ok(self.index.dataset().vector_len(id))
    }

    #[staticmethod]
    pub fn load(index_path: &str) -> PyResult<PySeismicIndex> {
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let index = bincode::deserialize::<SeismicIndex<f16>>(&serialized).unwrap();
        Ok(PySeismicIndex {
            index
        })
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
        self.index.inverted_index().knn().unwrap().serialize(path).unwrap();
    }

    pub fn load_knn(&mut self, knn_path: &str, nknn: usize) {
        let knn = Knn::new_from_serialized(knn_path, nknn);
        self.index.add_knn(knn);
    }

    // Load an index, add a pre computed knn-graph to it and return the new index
    #[staticmethod]
    pub fn load_index_knn(index_path: &str, knn_path: &str, nknn: usize) -> PyResult<PySeismicIndex> {
        println!("Loading ... {}", index_path);
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let mut index = bincode::deserialize::<SeismicIndex<f16>>(&serialized).unwrap();
        
        let knn = Knn::new_from_path(knn_path.to_string(), nknn);

        index.add_knn(knn);

        index.print_space_usage_byte();

        Ok(PySeismicIndex { index })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (json_path, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, nknn=0, knn_path=None, batched_indexing=None))]
    pub fn build(
        json_path: &str,
        n_postings: usize,
        centroid_fraction: f32,
        min_cluster_size: usize,
        summary_energy: f32,
        nknn: usize,
        knn_path: Option<String>,
        batched_indexing: Option<usize>,

        //TODO: missing token_to_id_mapping
    ) -> PyResult<PySeismicIndex> {
        let knn_config = KnnConfiguration::new(nknn, knn_path);


        //ATTENZIONE: sto usando i valori definiti di default: non considero fixed pruning, fixed block, 
        //            altri alg di clustering, fixed summary

        //max_fraction e doc_cut hardcoded, meglio definirli in rust e fare riferimento a parametro hardcoded
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
        
        let index = SeismicIndex::from_json(&json_path.to_owned(), config, None);
        Ok(PySeismicIndex { index })
    }

    pub fn search<'py>(
        &self,
        query_id: String,
        query_components: PyReadonlyArrayDyn<'py, PyFixedString<MAX_TOKEN_LEN>>,
        query_values: PyReadonlyArrayDyn<'py, f32>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool
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

    #[allow(clippy::too_many_arguments)]
    pub fn batch_search<'py>(
        &self,
        queries_ids: PyReadonlyArrayDyn<'py, PyFixedString<MAX_TOKEN_LEN>>,
        query_components: PyReadonlyArrayDyn<'py, PyFixedString<MAX_TOKEN_LEN>>,
        query_values: PyReadonlyArrayDyn<'py, f32>,
        offsets: PyReadonlyArrayDyn<'py, usize>,
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
        //let queries = SparseDataset::<f32>::read_bin_file(query_path).unwrap();
        let mut results = Vec::with_capacity(1000);
        for (query_id, w) in queries_ids
            .to_vec()
            .unwrap()
            .iter()
            .zip(offsets.to_vec().unwrap().windows(2))
        {
            let qc = query_components.as_slice().unwrap()[w[0]..w[1]]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>();
            let qv: Vec<f32> = query_values.as_slice().unwrap()[w[0]..w[1]]
                .iter()
                .map(|x| *x)
                .collect();

            results.push(self.index.search(
                &query_id.to_string(),
                &qc,
                &qv,
                k,
                query_cut,
                heap_factor,
                n_knn,
                sorted,
            ))
        }
        results
    }
}