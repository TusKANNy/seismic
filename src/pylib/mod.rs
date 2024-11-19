use crate::inverted_index::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, PruningStrategy, SummarizationStrategy, Knn, KnnConfiguration
};
use crate::{InvertedIndex, SparseDataset};
use half::f16;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;

#[pyclass]
pub struct PySeismicIndex {
    inverted_index: InvertedIndex<f16>,
}

#[pymethods]
impl PySeismicIndex {

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
    pub fn get_is_empty(&self) -> PyResult<bool> {
        Ok(self.inverted_index.is_empty())
    }

    pub fn print_space_usage_byte(&self) {
        self.inverted_index.print_space_usage_byte();
    }

    #[staticmethod]
    pub fn load(index_path: &str) -> PyResult<PySeismicIndex> {
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let inverted_index = bincode::deserialize::<InvertedIndex<f16>>(&serialized).unwrap();
        Ok(PySeismicIndex { inverted_index })
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

    // Load an index, add a pre computed knn-graph to it and return the new index
    #[staticmethod]
    pub fn load_index_knn(index_path: &str, knn_path: &str, nknn: usize) -> PyResult<PySeismicIndex> {
        println!("Loading ... {}", index_path);
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let mut inverted_index = bincode::deserialize::<InvertedIndex<f16>>(&serialized).unwrap();
        
        let knn = Knn::new_from_path(knn_path.to_string(), nknn);

        inverted_index.add_knn(knn);

        inverted_index.print_space_usage_byte();

        Ok(PySeismicIndex { inverted_index})
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_file, n_postings, centroid_fraction, min_cluster_size, summary_energy, nknn=0, knn_path=None, batched_indexing=None))]
    pub fn build(
        input_file: &str,
        n_postings: usize,
        centroid_fraction: f32,
        min_cluster_size: usize,
        summary_energy: f32,
        nknn: usize,
        knn_path: Option<String>,
        batched_indexing: Option<usize>,
    ) -> PyResult<PySeismicIndex> {
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
        Ok(PySeismicIndex { inverted_index })
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
