macro_rules! impl_seismic_dataset {
    ($rust_name:ident, $py_name:literal, $Key:ty) => {
        /// A Python wrapper around an in-memory sparse dataset.
        ///
        /// This class provides a way to build a dataset of sparse vectors
        /// (documents) in memory before indexing them with `SeismicIndex`.
        ///
        /// Each document is represented as a set of (token, value) pairs, where
        /// tokens are strings and values are float32.
        ///
        #[derive(Clone)]
        #[pyclass(name = $py_name)]
        pub struct $rust_name {
            dataset: Dataset<$Key, f16>,
        }

        #[pymethods]
        impl $rust_name {
            /// Create a new, empty SeismicDataset.
            ///
            /// This method returns an empty dataset ready to be populated
            /// with documents using `add_document()`.
            ///
            /// Returns:
            ///     SeismicDataset: A new, empty dataset instance.
            ///
            /// Example:
            ///     >>> dataset = seismic.SeismicDataset()
            #[new]
            fn new() -> Self {
                $rust_name {
                    dataset: Dataset::new(),
                }
            }

            /// Get the number of documents currently stored in the dataset.
            ///
            /// This method returns the total number of documents (sparse vectors)
            /// that have been added so far using `add_document()`.
            ///
            /// Returns:
            ///     int: The number of documents in the dataset.
            ///
            /// Example:
            ///     >>> dataset.len
            ///     3
            #[getter]
            pub fn get_len(&self) -> PyResult<usize> {
                Ok(self.dataset.len())
            }

            /// Add a sparse document to the dataset.
            ///
            /// This method adds a document identified by `doc_id`, using a list of
            /// tokens and their corresponding values. Each document is stored as a
            /// sparse vector of (token, value) pairs.
            ///
            /// Args:
            ///     doc_id (str): A unique identifier for the document.
            ///     tokens (ndarray[str]): Array of tokens (as strings).
            ///     values (ndarray[float32]): Array of corresponding values (same length as tokens).
            ///
            /// Example:
            ///     >>> string_type = seismic.get_seismic_string()
            ///     >>> dataset.add_document("doc42", np.array(["a", "b"], dtype=string_type), np.array([0.5, 1.0]))
            #[pyo3(signature = (doc_id, tokens, values))]
            #[pyo3(text_signature = "(self, doc_id, tokens, values)")]
            pub fn add_document(
                &mut self,
                doc_id: &str,
                tokens: PyReadonlyArrayDyn<'_, PyFixedUnicode<MAX_TOKEN_LEN>>,
                values: PyReadonlyArrayDyn<'_, f32>,
            ) {
                self.dataset.add_document(
                    doc_id.to_string(),
                    &tokens
                        .to_vec()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<String>>(),
                    &values.to_vec().unwrap(),
                );
            }
        }
    };
}
