use std::collections::HashMap;

use crate::{DataType, SparseDatasetMut};

pub struct SeismicDataset<T>
where
    T: DataType,
{
    sparse_dataset: SparseDatasetMut<T>,
    document_mapping: Vec<String>,
    token_to_id_map: HashMap<String, usize>,
}

impl<T> SeismicDataset<T>
where
    T: PartialOrd + DataType,
{
    pub fn new() -> Self
    where
        T: DataType + PartialOrd,
    {
        let sparse_dataset = SparseDatasetMut::new();
        let document_mapping = Vec::<String>::new();
        let token_to_id_map = HashMap::<String, usize>::new();
        Self {
            sparse_dataset,
            document_mapping: document_mapping,
            token_to_id_map,
        }
    }

    pub fn sparse_dataset(self) -> SparseDatasetMut<T> {
        self.sparse_dataset
    }

    pub fn document_mapping(&self) -> &Vec<String> {
        &self.document_mapping
    }

    pub fn token_to_id_map(&self) -> &HashMap<String, usize> {
        &self.token_to_id_map
    }

    pub fn len(&self) -> usize {
        self.sparse_dataset.len()
    }

    pub fn add_document(&mut self, id: String, tokens: &[String], values: &[f32]) {
        self.document_mapping.push(id);
        let mut components = Vec::<u16>::with_capacity(tokens.len());
        for c in tokens.iter() {
            let next_token_id = self.token_to_id_map.len(); // self.token_to_id_map.len() cannot be borrowed both mut and non mut
            components.push(
                *self
                    .token_to_id_map
                    .entry(c.to_string())
                    .or_insert_with(|| next_token_id) as u16,
            );

            self.sparse_dataset.push(
                &components,
                &values
                    .iter()
                    .map(|x| T::from_f32(*x).unwrap())
                    .collect::<Vec<T>>(),
            );
        }
    }
}
