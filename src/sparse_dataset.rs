//! This module provides structs to represent sparse datasets.
//!
//! A **sparse vector** in a `dim`-dimensional space consists of two sequences: components,
//! which are distinct values in the range [0, `dim`), and their corresponding values of type `T`.
//! The type `T` is typically expected to be a float type such as `f16`, `f32`, or `f64`.
//!The components are of a generic type C which must implement the [ComponentType] trait.
//!
//! A dataset is a collection of sparse vectors. This module provides two representations
//! of a sparse dataset: a mutable [`SparseDatasetMut`] and its immutable counterpart [`SparseDataset`].
//! Conversion between the two representations is straightforward, as both implement the [`From`] trait.

use rayon::iter::plumbing::ProducerCallback;
use serde::{Deserialize, Serialize};

use std::fmt::Debug;
// Reading files
use std::fs::File;
use std::io::{BufReader, Read, Result as IoResult};
use std::iter::Zip;
use std::ops::Range;
use std::path::Path;

use rayon::iter::plumbing::{bridge, Consumer, Producer, UnindexedConsumer};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::topk_selectors::{HeapFaiss, OnlineTopKSelector};
use crate::utils::{conditionally_densify, prefetch_read_NTA};
use crate::{ComponentType, SpaceUsage, ValueType};

// Implementation of a (immutable) sparse dataset.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparseDataset<C, T>
where
    T: ValueType,
{
    n_vecs: usize,
    d: usize,
    offsets: Box<[usize]>,
    components: Box<[C]>,
    values: Box<[T]>,
}

impl<C, T> SparseDataset<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    /// Retrieves the components and values of the sparse vector at the specified index.
    ///
    /// This method returns a tuple containing slices of components and values
    /// of the sparse vector located at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the specified index is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// let (components, values) = dataset.get(1);
    /// assert_eq!(components, &[1, 3]);
    /// assert_eq!(values, &[4.0, 5.0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn get(&self, id: usize) -> (&[C], &[T]) {
        //assert!(id < self.n_vecs, "The id is out of range"); this check is performed by range method

        let v_components = &self.components[Self::vector_range(&self.offsets, id)];
        let v_values = &self.values[Self::vector_range(&self.offsets, id)];

        (v_components, v_values)
    }

    /// Returns a vector of the dataset at the specified `offset` and `len`.
    ///
    /// This method returns slices of components and values for the dataset starting at the specified `offset`
    /// and with the specified `len`.
    ///
    /// This method is needed by [`InvertedIndex`] which often already knows the offset of the required vector. This speeds up the access.
    ///
    /// # Panics
    ///
    /// Panics if the `offset` + `len` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// let (components, values) = dataset.get_with_offset(3, 2);
    /// assert_eq!(components, &[1, 3]);
    /// assert_eq!(values, &[4.0, 5.0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn get_with_offset(&self, offset: usize, len: usize) -> (&[C], &[T]) {
        assert!(
            offset + len <= self.components.len(),
            "The id is out of range"
        );

        let v_components = &self.components[offset..offset + len];
        let v_values = &self.values[offset..offset + len];

        (v_components, v_values)
    }

    // Returns the range of positions of the slice with the given `id`.
    // It take `offsets` as a parameter to be shared with SparseDatasetMut.
    //
    // ### Panics
    // Panics if the `id` is out of range.
    #[inline]
    #[must_use]
    fn vector_range(offsets: &[usize], id: usize) -> Range<usize> {
        assert!(id <= offsets.len(), "{id} is out of range");

        // Safety: safe accesses due to the check above
        unsafe {
            Range {
                start: *offsets.get_unchecked(id),
                end: *offsets.get_unchecked(id + 1),
            }
        }
    }

    /// Prefetches the components and values of specified vectors into the CPU cache.
    ///
    /// This method prefetches the components and values of the vectors specified by their indices
    /// into the CPU cache, which can improve performance by reducing cache misses during subsequent accesses.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// // Prefetch components and values of vectors 0 and 1
    /// dataset.prefetch_vecs(&[0, 1]);
    /// ```
    #[inline]
    pub fn prefetch_vecs(&self, vecs: &[usize]) {
        for &vec_id in vecs.iter() {
            let start = self.offsets[vec_id];
            let end = self.offsets[vec_id + 1];

            for i in (start..end).step_by(512 / (std::mem::size_of::<C>() * 8)) {
                prefetch_read_NTA(&self.components, i);
            }

            for i in (start..end).step_by(512 / (std::mem::size_of::<T>() * 8)) {
                prefetch_read_NTA(&self.values, i);
            }
        }
    }

    /// Prefetches the components and values of a vector with the specified offset and length into the CPU cache.
    ///
    /// This method prefetches the components and values of a vector starting at the specified `offset``
    /// and with the specified length `len` into the CPU cache, which can improve performance by reducing
    /// cache misses during subsequent accesses.
    ///
    /// # Parameters
    ///
    /// * `offset`: The starting index of the vector to prefetch.
    /// * `len`: The length of the vector to prefetch.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// // Prefetch components and values of the vector starting at index 1 with length 3
    /// dataset.prefetch_vec_with_offset(1, 3);
    /// ```
    #[inline]
    pub fn prefetch_vec_with_offset(&self, offset: usize, len: usize) {
        let end = offset + len;

        for i in (offset..end).step_by(512 / (std::mem::size_of::<u16>() * 8)) {
            prefetch_read_NTA(&self.components, i);
        }

        for i in (offset..end).step_by(512 / (std::mem::size_of::<T>() * 8)) {
            prefetch_read_NTA(&self.values, i);
        }
    }

    /// Returns the offset of the vector with the specified index.
    ///
    /// This method returns the offset of the vector with the specified index in the dataset.
    /// The offset represents the starting index of the vector within the internal storage.
    ///
    /// # Panics
    /// Panics if the `id` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.vector_offset(1), 3); // Offset of the vector with index 1
    /// ```
    #[must_use]
    #[inline]
    pub fn vector_offset(&self, id: usize) -> usize {
        assert!(id < self.n_vecs, "the id is out of range");

        self.offsets[id]
    }

    /// Converts a `SparseDataset<C, T>` into a `SparseDataset<C, Q>`.
    ///
    /// This conversion is useful when you want to change the value type of the dataset.
    ///
    /// # Example
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// use half::f16;
    /// let new_dataset: SparseDataset<u16, f16> = dataset.quantize();
    /// ```
    #[must_use]
    pub fn quantize<Q: ValueType>(self) -> SparseDataset<C, Q> {
        let values: Vec<_> = self
            .values
            .iter()
            .map(|&v| {
                let v: f32 = v.to_f32();
                Q::from_f32(v)
            })
            .collect();

        SparseDataset::<C, Q> {
            d: self.d,
            offsets: self.offsets,
            n_vecs: self.n_vecs,
            components: self.components,
            values: values.into_boxed_slice(),
        }
    }

    /// Returns the length of the vector with the specified index.
    ///
    /// This method returns the length of the vector with the specified index in the dataset.
    /// The length represents the number of non-zero components in the vector.
    ///
    /// # Panics
    ///
    /// Panics if the specified index is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.vector_len(1), 2); // Length of the vector with index 1
    /// ```
    #[must_use]
    #[inline]
    pub fn vector_len(&self, id: usize) -> usize {
        assert!(id < self.n_vecs, "The id is out of range");

        self.offsets[id + 1] - self.offsets[id]
    }

    /// Performs a brute-force search to find the K-nearest neighbors (KNN) of the queried vector.
    ///
    /// This method scans the entire dataset to find the K-nearest neighbors of the queried vector.
    /// It computes the *dot product* between the queried vector and each vector in the dataset and returns
    /// the indices of the K-nearest neighbors along with their distances.
    ///
    /// # Parameters
    ///
    /// * `q_components`: The components of the queried vector.
    /// * `q_values`: The values corresponding to the components of the queried vector.
    /// * `k`: The number of nearest neighbors to find.
    ///
    /// # Returns
    ///
    /// A vector containing tuples of distances and indices of the K-nearest neighbors, sorted by decreasing distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// let query_components = &[0, 2];
    /// let query_values = &[1.0, 1.0];
    /// let k = 2;
    ///
    /// let knn = dataset.search(query_components, query_values, k);
    ///
    /// assert_eq!(knn, vec![(4.0, 2), (3.0, 0)]);
    /// ```
    #[must_use]
    #[inline]
    pub fn search(&self, q_components: &[C], q_values: &[f32], k: usize) -> Vec<(f32, usize)> {
        let dense_query = conditionally_densify(q_components, q_values, self.dim());
        let distances: Vec<_> = (0..self.n_vecs)
            .map(|id| {
                let v_components = &self.components[Self::vector_range(&self.offsets, id)];
                let v_values = &self.values[Self::vector_range(&self.offsets, id)];
                -1.0 * C::compute_dot_product(
                    dense_query.as_deref(),
                    q_components,
                    q_values,
                    v_components,
                    v_values,
                )
            })
            .collect();

        let mut heap = HeapFaiss::new(k);
        heap.extend(&distances);

        heap.topk().into_iter().map(|(d, i)| (d.abs(), i)).collect()
    }

    /// Returns an iterator over the vectors of the dataset.
    ///
    /// This method returns an iterator that yields references to each vector in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.clone().into_iter().collect();
    ///
    /// for ((c0, v0), (c1,v1)) in dataset.iter().zip(data.iter()) {
    ///     assert_eq!(c0, c1);
    ///     assert_eq!(v0, v1);
    /// }
    /// ```
    pub fn iter(&self) -> SparseDatasetIter<C, T> {
        SparseDatasetIter::new(self)
    }

    /// Returns an iterator over the sparse vector with id `vec_id`.
    ///
    /// # Panics
    /// Panics if the `vec_id` is out of bounds.
    pub fn iter_vector(
        &self,
        vec_id: usize,
    ) -> Zip<std::slice::Iter<'_, C>, std::slice::Iter<'_, T>> {
        assert!(vec_id < self.n_vecs, "The id {vec_id} is out of range");

        let start = self.offsets[vec_id];
        let end = self.offsets[vec_id + 1];

        let v_components = &self.components[start..end];
        let v_values = &self.values[start..end];

        v_components.iter().zip(v_values)
    }

    /// Returns the number of vectors in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_vecs
    }

    /// Checks if the dataset is empty.
    ///     
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let mut dataset = SparseDataset::<u16, f32>::default();
    ///
    /// assert!(dataset.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of components of the dataset, i.e., it returns one plus the ID of the largest component.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.dim(), 5); // Largest component ID is 4, so dim() returns 5
    /// ```
    #[must_use]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Returns the number of non-zero components in the dataset.
    ///
    /// This function returns the total count of non-zero components across all vectors
    /// currently stored in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.components.len()
    }

    /// Converts the `offset` of a vector within the dataset to its id, i.e., the position
    /// of the vector within the dataset.
    ///
    /// # Panics
    /// Panics if the `offset` is not the first postion of a vector in the dataset.
    #[must_use]
    #[inline]
    pub fn offset_to_id(&self, offset: usize) -> usize {
        self.offsets.binary_search(&offset).unwrap()
    }

    #[must_use]
    #[inline]
    pub fn id_to_offset(&self, id: usize) -> usize {
        assert!(id < self.n_vecs, "The id is out of range");
        self.offsets[id]
    }

    #[must_use]
    #[inline]
    pub fn id_to_offset_len(&self, id: usize) -> (usize, usize) {
        assert!(id < self.n_vecs, "The id is out of range");
        (self.offsets[id], self.offsets[id + 1] - self.offsets[id])
    }

    #[must_use]
    #[inline]
    pub fn id_to_encoded_offset(&self, id: usize) -> usize {
        assert!(id < self.n_vecs, "The id is out of range");
        self.offsets[id] / 2
    }

    // The format of this binary file is the following.
    // Number of vectors n_vecs qin 4 bytes, follows n_vecs sparse vectors.
    // For each vector we encode:
    //  - Its length in 4 bytes;
    //  - A sorted sequence of n components in 4 bytes each;
    //  - Corresponding n scores in 4 bytes each
    // Writing in Python is a follows
    // ```python
    // import struct
    // def sparse_vectors_to_binary_file(filename, term_id):
    // # A binary sequence is a sequence of integers prefixed by its length,
    // # where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.
    // # Followed by a sequence of f32, with the same length
    // def write_binary_sequence(lst_pairs, file):
    //     file.write((len(lst_pairs)).to_bytes(4, byteorder='little', signed=False))
    //     for v in lst_pairs:
    //         file.write((v[0]).to_bytes(4, byteorder='little', signed=False))
    //     for v in lst_pairs:
    //         value = v[1]
    //         ba = bytearray(struct.pack("f", value))
    //         file.write(ba)
    // with open(filename, "wb") as fout:
    //     fout.write((len(term_id)).to_bytes(4, byteorder='little', signed=False))
    //     for d in term_id:
    //         lst = sorted(list(d.items()))
    //         write_binary_sequence(lst, fout)
    // ````
    /// # Panics
    /// Panics if a component cannot be converted to type `C`.
    pub fn read_bin_file(fname: &str) -> IoResult<SparseDataset<C, f32>>
    where
        C: ComponentType,
        //        <C as TryFrom<u32>>::Error: std::fmt::Debug,
    {
        Self::read_bin_file_limit(fname, None)
    }

    pub fn read_bin_file_limit(fname: &str, limit: Option<usize>) -> IoResult<SparseDataset<C, f32>>
    where
        C: ComponentType,
        //        <C as TryFrom<u32>>::Error: std::fmt::Debug,
    {
        let path = Path::new(fname);
        let f = File::open(path)?;
        //let f_size = f.metadata().unwrap().len() as usize;

        let mut br = BufReader::new(f);

        let mut buffer_d = [0u8; std::mem::size_of::<u32>()];
        let mut buffer = [0u8; std::mem::size_of::<f32>()];

        br.read_exact(&mut buffer_d)?;
        let mut n_vecs = u32::from_le_bytes(buffer_d) as usize;

        if let Some(n) = limit {
            n_vecs = n.min(n_vecs);
        }

        let mut data = SparseDatasetMut::<C, f32>::default();
        for _ in 0..n_vecs {
            br.read_exact(&mut buffer_d)?;
            let n = u32::from_le_bytes(buffer_d) as usize;

            let mut components = Vec::with_capacity(n);
            let mut values = Vec::<f32>::with_capacity(n);

            for _ in 0..n {
                br.read_exact(&mut buffer_d)?;
                let c = C::from_u32(u32::from_le_bytes(buffer_d)).unwrap();
                //.try_into()
                //.expect("Failed to convert to type C (Component Type).");
                components.push(c);
            }
            for _ in 0..n {
                br.read_exact(&mut buffer)?;
                let v = f32::from_le_bytes(buffer);
                values.push(v);
            }

            data.push(&components, &values);
        }

        Ok(data.into())
    }
}

/// A mutable representation of a sparse dataset.
///
/// This struct provides functionality for manipulating a sparse dataset with mutable access.
/// It stores sparse vectors in the dataset, where each vector consists of components
/// and their corresponding values of type `T`. The type `T` must implement the `SpaceUsage` trait,
/// allowing for efficient memory usage calculations, and it must also be `Copy`.
///
/// # Examples
///
/// ```
/// use seismic::SparseDatasetMut;
///
/// // Create a new empty dataset
/// let mut dataset = SparseDatasetMut::<u16, f32>::default();
/// ```
///
///
#[derive(Debug, Clone)]
pub struct SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: SpaceUsage + ValueType,
{
    d: usize,
    offsets: Vec<usize>,
    components: Vec<C>,
    values: Vec<T>,
}

impl<C, T> Default for SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: SpaceUsage + ValueType,
{
    /// Constructs a new, empty mutable sparse dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let dataset: SparseDatasetMut<u16, f64> = SparseDatasetMut::default();
    ///
    /// assert!(dataset.is_empty());
    /// ```
    fn default() -> Self {
        Self {
            d: 0,
            offsets: vec![0; 1],
            components: Vec::new(),
            values: Vec::new(),
        }
    }
}

impl<C, T> SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: SpaceUsage + ValueType,
{
    /// Constructs a new, empty mutable sparse dataset.
    ///
    /// # Example
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let mut dataset = SparseDatasetMut::<u16, f32>::default();
    ///
    /// assert!(dataset.is_empty());
    /// ```
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDatasetMut<T>`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieves the components and values of the sparse vector at the specified index.
    ///
    /// This method returns a tuple containing slices of components and values
    /// of the sparse vector located at the given index.
    ///
    /// # Parameters
    ///
    /// * `id`: The index of the sparse vector to retrieve.
    ///
    /// # Returns
    ///
    /// A tuple containing slices of components and values of the sparse vector.
    ///
    /// # Panics
    ///
    /// Panics if the specified index is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    ///
    /// let (components, values) = dataset.get(1);
    /// assert_eq!(components, &[1, 3]);
    /// assert_eq!(values, &[4.0, 5.0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn get(&self, id: usize) -> (&[C], &[T]) {
        //assert!(id < self.n_vecs, "The id is out of range"); check already done by range

        let v_components = &self.components[SparseDataset::<C, T>::vector_range(&self.offsets, id)];
        let v_values = &self.values[SparseDataset::<C, T>::vector_range(&self.offsets, id)];

        (v_components, v_values)
    }

    //TODO: we need to change this into push_iterator.
    pub fn push_pairs(&mut self, pairs: &[(C, T)]) {
        assert!(
            pairs.windows(2).all(|w| w[0].0 < w[1].0),
            "Components must be given in sorted order"
        );

        self.offsets
            .push(*self.offsets.last().unwrap() + pairs.len());

        if pairs.is_empty() {
            return;
        }

        if pairs.last().unwrap().0.as_() >= self.d {
            self.d = pairs.last().unwrap().0.as_() + 1;
        }

        self.components.extend(pairs.iter().map(|(c, _)| c));
        self.values.extend(pairs.iter().map(|(_, v)| v));
    }

    /// Adds a new sparse vector to the dataset.
    ///
    /// The `components` parameter is assumed to be a strictly increasing sequence
    /// representing the indices of non-zero values in the vector, and `values`
    /// holds the corresponding values. Both `components` and `values` must have
    /// the same size and cannot be empty. Additionally, `components` must be sorted.
    ///
    /// # Parameters
    ///
    /// * `components`: A slice containing the indices of non-zero values in the vector.
    /// * `values`: A slice containing the corresponding values for each index in `components`.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    ///
    /// * The sizes of `components` and `values` are different.
    /// * The size of either `components` or `values` is 0.
    /// * `components` is not sorted in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let mut dataset = SparseDatasetMut::<u16, f32>::default();
    /// dataset.push(&[0, 2, 4], &[1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(dataset.len(), 1);
    /// assert_eq!(dataset.dim(), 5);
    /// assert_eq!(dataset.nnz(), 3);
    /// ```
    pub fn push(&mut self, components: &[C], values: &[T]) {
        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );
        // a vector can be empty now
        // assert!(!components.is_empty());
        assert!(
            components.windows(2).all(|w| w[0] <= w[1]),
            "Components must be given in sorted order"
        );

        self.offsets
            .push(*self.offsets.last().unwrap() + components.len());

        if components.is_empty() {
            return;
        }

        if (*components.last().unwrap()).as_() >= self.d {
            self.d = (*components.last().unwrap()).as_() + 1;
        }

        self.components.extend(components);
        self.values.extend(values);
    }

    /// Returns the length of the vector with the specified index.
    ///
    /// This method returns the length of the vector with the specified index in the dataset.
    /// The length represents the number of non-zero components in the vector.
    ///
    /// # Panics
    ///
    /// Panics if the specified index is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.vector_len(1), 2); // Length of the vector with index 1
    /// ```
    #[must_use]
    #[inline]
    pub fn vector_len(&self, id: usize) -> usize {
        assert!(id < self.offsets.len() - 1, "The id is out of range");

        self.offsets[id + 1] - self.offsets[id]
    }

    /// Returns the number of vectors in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let mut dataset = SparseDatasetMut::<u16, f32>::default();
    /// dataset.push(&[0, 2, 4], &[1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(dataset.len(), 1);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Checks if the dataset is empty.
    ///     
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let mut dataset = SparseDatasetMut::<u16, f32>::default();
    /// dataset.push(&[0, 2, 4], &[1.0, 2.0, 3.0]);
    ///
    /// assert!(!dataset.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of components of the dataset, i.e., it returns one plus the ID of the largest component.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let mut dataset = SparseDatasetMut::<u16, f32>::new();
    ///
    /// dataset.push(&[0, 2, 4], &[1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(dataset.dim(), 5); // Largest component ID is 4, so dim() returns 5
    /// ```
    #[must_use]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Returns the number of non-zero components in the dataset.
    ///
    /// This function returns the total count of non-zero components across all vectors
    /// currently stored in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.components.len()
    }

    /// Returns an iterator over the vectors of the dataset.
    ///
    /// This method returns an iterator that yields references to each vector in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.clone().into_iter().collect();
    ///
    /// for ((c0, v0), (c1,v1)) in dataset.iter().zip(data.iter()) {
    ///     assert_eq!(c0, c1);
    ///     assert_eq!(v0, v1);
    /// }
    /// ```
    pub fn iter(&self) -> SparseDatasetIter<C, T> {
        SparseDatasetIter::new_with_mut(self)
    }

    /// Returns an iterator over the sparse vector with the specified id.
    ///
    /// This method returns an iterator that yields pairs of components and values for the sparse vector
    /// with the specified id.
    ///
    /// # Parameters
    ///
    /// * `vec_id`: The id of the sparse vector to iterate over.
    ///
    /// # Returns
    ///
    /// An iterator over the components and values of the sparse vector.
    ///
    /// # Panics
    ///
    /// Panics if the specified `vec_id` is out of bounds.
    pub fn iter_vector(
        &self,
        vec_id: usize,
    ) -> std::iter::Zip<std::slice::Iter<'_, C>, std::slice::Iter<'_, T>> {
        assert!(vec_id < self.len(), "The id {} is out of range", vec_id);

        let start = self.offsets[vec_id];
        let end = self.offsets[vec_id + 1];

        let v_components = &self.components[start..end];
        let v_values = &self.values[start..end];

        v_components.iter().zip(v_values)
    }
}

impl<C, T> FromIterator<(Vec<C>, Vec<T>)> for SparseDataset<C, T>
where
    C: ComponentType,
    T: SpaceUsage + ValueType,
{
    /// Constructs a `SparseDataset<C, T>` from an iterator over pairs of vectors.
    ///
    /// This function consumes the provided iterator and constructs a new `SparseDataset<T>`.
    /// Each pair in the iterator represents a pair of vectors, where the first vector contains
    /// the components and the second vector contains their corresponding values.
    ///
    /// # Parameters
    ///
    /// * `iter`: An iterator over pairs of vectors `(vec[C], vec[T])`.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDataset<C, T>` populated with the pairs from the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Vec<C>, Vec<T>)>,
    {
        let mut dataset = SparseDatasetMut::new();

        for (components, values) in iter {
            dataset.push(&components, &values);
        }

        dataset.into()
    }
}

impl<C, T> FromIterator<(Vec<C>, Vec<T>)> for SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: SpaceUsage + ValueType,
{
    /// Constructs a `SparseDatasetMut<C, T>` from an iterator over pairs of vectors.
    ///
    /// This function consumes the provided iterator and constructs a new `SparseDataset<T>`.
    /// Each pair in the iterator represents a pair of vectors, where the first vector contains
    /// the components and the second vector contains their corresponding values.
    ///
    /// # Parameters
    ///
    /// * `iter`: An iterator over pairs of vectors `(vec[C], vec[T])`.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDatasetMut<C, T>` populated with the pairs from the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Vec<C>, Vec<T>)>,
    {
        let mut dataset = SparseDatasetMut::new();

        for (components, values) in iter {
            dataset.push(&components, &values);
        }

        dataset
    }
}

impl<'a, C, T> FromIterator<(&'a [C], &'a [T])> for SparseDataset<C, T>
where
    C: ComponentType,
    T: ValueType + SpaceUsage,
{
    /// Constructs a `SparseDataset<C, T>` from an iterator over pairs of slices.
    ///
    /// This function consumes the provided iterator and constructs a new `SparseDataset<T>`.
    /// Each pair in the iterator represents a pair of slices, where the first slice contains
    /// the components and the second slice contains their corresponding values.
    ///
    /// # Parameters
    ///
    /// * `iter`: An iterator over pairs of slices `(&[C], &[T])`.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDataset<C, T>` populated with the pairs from the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = &[(vec![0, 2, 4], vec![1.0, 2.0, 3.0]), (vec![1, 3], vec![4.0, 5.0]), (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])];
    /// let dataset: SparseDataset<u16, f32> = data.iter().map(|(c, v)| (c.as_slice(), v.as_slice()) ).collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (&'a [C], &'a [T])>,
    {
        let mut dataset = SparseDatasetMut::new();

        for (components, values) in iter {
            dataset.push(components, values);
        }

        dataset.into()
    }
}

impl<'a, C, T> FromIterator<(&'a [C], &'a [T])> for SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: SpaceUsage + ValueType,
{
    /// Constructs a `SparseDatasetMut<C, T>` from an iterator over pairs of slices.
    ///
    /// This function consumes the provided iterator and constructs a new `SparseDatasetMut<T>`.
    /// Each pair in the iterator represents a pair of slices, where the first slice contains
    /// the components and the second slice contains their corresponding values.
    ///
    /// # Parameters
    ///
    /// * `iter`: An iterator over pairs of slices `(&[C], &[T])`.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDatasetMut<C, T>` populated with the pairs from the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = &[(vec![0, 2, 4], vec![1.0, 2.0, 3.0]), (vec![1, 3], vec![4.0, 5.0]), (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])];
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().map(|(c, v)| (c.as_slice(), v.as_slice())).collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (&'a [C], &'a [T])>,
    {
        let mut dataset = SparseDatasetMut::<C, T>::new();

        for (components, values) in iter {
            dataset.push(components, values);
        }

        dataset
    }
}

impl<C, T> From<SparseDatasetMut<C, T>> for SparseDataset<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    /// Converts a mutable sparse dataset into an immutable one.
    ///
    /// This function consumes the provided `SparseDatasetMut<T>` and produces
    /// a corresponding immutable `SparseDataset<T>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the mutable dataset
    /// to the immutable one.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::{SparseDatasetMut, SparseDataset};
    ///
    /// let mut mutable_dataset = SparseDatasetMut::<u16, f32>::new();
    /// // Populate mutable dataset...
    /// mutable_dataset.push(&[0, 2, 4],    &[1.0, 2.0, 3.0]);
    /// mutable_dataset.push(&[1, 3],       &[4.0, 5.0]);
    /// mutable_dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// let immutable_dataset: SparseDataset<u16, f32> = mutable_dataset.into();
    ///
    /// assert_eq!(immutable_dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from(dataset: SparseDatasetMut<C, T>) -> Self {
        Self {
            n_vecs: dataset.offsets.len() - 1,
            d: dataset.d,
            offsets: dataset.offsets.into_boxed_slice(),
            components: dataset.components.into_boxed_slice(),
            values: dataset.values.into_boxed_slice(),
        }
    }
}

impl<C, T> AsRef<SparseDataset<C, T>> for SparseDataset<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    #[inline]
    fn as_ref(&self) -> &SparseDataset<C, T> {
        self
    }
}

impl<C, T> From<SparseDataset<C, T>> for SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    /// Converts an immutable sparse dataset into a mutable one.
    ///
    /// This function consumes the provided `SparseDataset<C, T>` and produces
    /// a corresponding mutable `SparseDatasetMut<C, T>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the immutable dataset
    /// to the mutable one.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::{SparseDatasetMut, SparseDataset};
    ///
    /// let mut mutable_dataset = SparseDatasetMut::<u16, f32>::new();
    /// // Populate mutable dataset...
    /// mutable_dataset.push(&[0, 2, 4],    &[1.0, 2.0, 3.0]);
    /// mutable_dataset.push(&[1, 3],       &[4.0, 5.0]);
    /// mutable_dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// let immutable_dataset: SparseDataset<u16, f32> = mutable_dataset.into();
    ///
    /// // Convert immutable dataset back to mutable
    /// let mut mutable_dataset_again: SparseDatasetMut<u16, f32> = immutable_dataset.into();
    ///
    /// mutable_dataset_again.push(&[1, 7], &[1.0, 3.0]);
    ///
    /// assert_eq!(mutable_dataset_again.nnz(), 11); // Total non-zero components across all vectors
    /// ```
    fn from(dataset: SparseDataset<C, T>) -> Self {
        Self {
            d: dataset.d,
            offsets: dataset.offsets.into(),
            components: dataset.components.into(),
            values: dataset.values.into(),
        }
    }
}

impl<'a, C, T> IntoParallelIterator for &'a SparseDataset<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    type Iter = ParSparseDatasetIter<'a, C, T>;
    type Item = (&'a [C], &'a [T]);

    fn into_par_iter(self) -> Self::Iter {
        ParSparseDatasetIter {
            last_offset: self.offsets[0],
            offsets: &self.offsets[1..],
            components: &self.components,
            values: &self.values,
        }
    }
}

impl<'a, C, T> IntoParallelIterator for &'a SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    type Iter = ParSparseDatasetIter<'a, C, T>;
    type Item = (&'a [C], &'a [T]);

    fn into_par_iter(self) -> Self::Iter {
        ParSparseDatasetIter {
            last_offset: self.offsets[0],
            offsets: &self.offsets[1..],
            components: &self.components,
            values: &self.values,
        }
    }
}

/// A struct to iterate over a sparse dataset. It assumes the dataset can be represented as a pair of slices.
#[derive(Clone)]
pub struct SparseDatasetIter<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [T],
}

impl<'a, C, T> SparseDatasetIter<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    #[inline]
    fn new(dataset: &'a SparseDataset<C, T>) -> Self {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets[1..],
            components: &dataset.components,
            values: &dataset.values,
        }
    }

    #[inline]
    fn new_with_mut(dataset: &'a SparseDatasetMut<C, T>) -> Self {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets[1..],
            components: &dataset.components,
            values: &dataset.values,
        }
    }
}

/// A struct to iterate over a sparse dataset in parallel.
/// It assumes the dataset can be represented as a pair of slices.
#[derive(Clone)]
pub struct ParSparseDatasetIter<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [T],
}

// impl<'a, T> ParSparseDatasetIter<'a, T>
// where
//     T: ValueType,
// {
//     #[inline]
//     fn new(dataset: &'a SparseDataset<T>) -> Self {
//         Self {
//             last_offset: 0,
//             offsets: &dataset.offsets[1..],
//             components: &dataset.components,
//             values: &dataset.values,
//         }
//     }
// }

impl<'a, C, T> Iterator for SparseDatasetIter<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    type Item = (&'a [C], &'a [T]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&next_offset, rest) = self.offsets.split_first()?;
        self.offsets = rest;

        let (cur_components, rest) = self.components.split_at(next_offset - self.last_offset);
        self.components = rest;

        let (cur_values, rest) = self.values.split_at(next_offset - self.last_offset);
        self.values = rest;

        self.last_offset = next_offset;

        Some((cur_components, cur_values))
    }
}

impl<'a, C, T> ParallelIterator for ParSparseDatasetIter<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    type Item = (&'a [C], &'a [T]);

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
    where
        CS: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.offsets.len())
    }
}

impl<C, T> IndexedParallelIterator for ParSparseDatasetIter<'_, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = SparseDatasetProducer::from(self);
        callback.callback(producer)
    }

    fn drive<CS: Consumer<Self::Item>>(self, consumer: CS) -> CS::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<C, T> ExactSizeIterator for SparseDatasetIter<'_, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<C, T> DoubleEndedIterator for SparseDatasetIter<'_, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    /// Retrieves the next vector from the end of the iterator.
    ///
    /// # Returns
    ///
    /// An option containing a tuple of slices representing the components and values of the next vector,
    /// or `None` if the end of the iterator is reached.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDataset;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDataset<u16, f32> = data.clone().into_iter().collect();
    ///
    /// let data_rev: Vec<_> =  dataset.iter().rev().collect();
    ///
    /// for ((c0,v0), (c1, v1)) in data.into_iter().zip(data_rev.into_iter().rev()) {
    ///     assert_eq!(c0.as_slice(), c1);
    ///     assert_eq!(v0.as_slice(), v1);
    /// }
    /// ```
    fn next_back(&mut self) -> Option<Self::Item> {
        let (&last_offset, rest) = self.offsets.split_last()?;
        self.offsets = rest;

        let next_offset = *self.offsets.last().unwrap_or(&self.last_offset);

        let len = last_offset - next_offset;

        let (rest, cur_components) = self.components.split_at(last_offset - len);
        self.components = rest;

        let (rest, cur_values) = self.values.split_at(last_offset - len);
        self.values = rest;

        Some((cur_components, cur_values))
    }
}
struct SparseDatasetProducer<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [T],
}

impl<'a, C, T> Producer for SparseDatasetProducer<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    type Item = (&'a [C], &'a [T]);
    type IntoIter = SparseDatasetIter<'a, C, T>;

    fn into_iter(self) -> Self::IntoIter {
        SparseDatasetIter {
            last_offset: self.last_offset,
            offsets: self.offsets,
            components: self.components,
            values: self.values,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left_last_offset = self.last_offset;

        let (left_offsets, right_offsets) = self.offsets.split_at(index);
        let right_last_offset = *left_offsets.last().unwrap();

        let (left_components, right_components) = self
            .components
            .split_at(right_last_offset - left_last_offset);
        let (left_values, right_values) =
            self.values.split_at(right_last_offset - left_last_offset);

        (
            SparseDatasetProducer {
                last_offset: left_last_offset,
                offsets: left_offsets,
                components: left_components,
                values: left_values,
            },
            SparseDatasetProducer {
                last_offset: right_last_offset,
                offsets: right_offsets,
                components: right_components,
                values: right_values,
            },
        )
    }
}

impl<'a, C, T> From<ParSparseDatasetIter<'a, C, T>> for SparseDatasetProducer<'a, C, T>
where
    C: ComponentType,
    T: ValueType,
{
    fn from(other: ParSparseDatasetIter<'a, C, T>) -> Self {
        Self {
            last_offset: other.last_offset,
            offsets: other.offsets,
            components: other.components,
            values: other.values,
        }
    }
}

impl<C, T> SpaceUsage for SparseDataset<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.n_vecs.space_usage_byte()
            + self.d.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.components.space_usage_byte()
            + self.values.space_usage_byte()
    }
}

impl<C, T> SpaceUsage for SparseDatasetMut<C, T>
where
    C: ComponentType,
    T: ValueType,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.d.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.components.space_usage_byte()
            + self.values.space_usage_byte()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test pushing empty vectors.
    #[test]
    fn test_empty_vectors() {
        let mut dataset = SparseDatasetMut::<u16, f32>::default();

        // Push a single vector
        dataset.push(&[0, 2, 4], &[1.0, 2.0, 3.0]);
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        // Push another vector
        let c = Vec::new();
        let v = Vec::new();

        dataset.push(&c, &v);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        dataset.push(&c, &v);
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        // Push a fourth vector
        dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.dim(), 5);
        assert_eq!(dataset.nnz(), 7);

        let dataset = SparseDataset::from(dataset);

        // Check the contents of the dataset
        let (components, values) = dataset.get(0);
        assert_eq!(components, &[0, 2, 4]);
        assert_eq!(values, &[1.0, 2.0, 3.0]);

        let (components, values) = dataset.get(1);
        assert!(components.is_empty());
        assert!(values.is_empty());

        let (components, values) = dataset.get(2);
        assert!(components.is_empty());
        assert!(values.is_empty());

        let (components, values) = dataset.get(3);
        assert_eq!(components, &[0, 1, 2, 3]);
        assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
    }

    // Test iteration (forward and backward) over the vectors of a collection.
    #[test]
    fn test_double_ended_iterator() {
        let size: usize = 13;

        let n_vecs = 10;
        let n = n_vecs * size;

        let components: Vec<_> = (0_u16..n as u16).collect();
        let values: Vec<_> = (0..n).map(|x| x as f32).collect();
        let mut dataset = SparseDatasetMut::<u16, f32>::default();

        let result: Vec<_> = components
            .chunks_exact(size)
            .zip(values.chunks_exact(size))
            .collect();

        for (c, v) in result.iter() {
            dataset.push(c, v);
        }

        let dataset = SparseDataset::from(dataset);

        // iter()
        let vec: Vec<_> = dataset.iter().collect();

        assert_eq!(vec, result);

        // reverse iter()
        let mut iter = dataset.iter();
        let mut vec = Vec::new();
        while let Some((c, v)) = iter.next_back() {
            vec.push((c, v));
        }
        vec.reverse();

        assert_eq!(vec, result);
    }
}
