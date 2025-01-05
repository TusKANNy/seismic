//! This module provides structs to represent sparse datasets.
//!
//! A **sparse vector** in a `dim`-dimensional space consists of two sequences: components,
//! which are distinct values in the range [0, `dim`), and their corresponding values of type `V`.
//! The type `V` is typically expected to be a float type such as `f16`, `f32`, or `f64`.
//! The components are of a generic type C which must implement the [ComponentType] trait.
//!
//! A dataset is a collection of sparse vectors. This module provides two representations
//! of a sparse dataset: a mutable [`SparseDatasetMut`] and its immutable counterpart [`SparseDataset`].
//! Conversion between the two representations is straightforward, as both implement the [`From`] trait.

use itertools::Itertools;
use rayon::iter::plumbing::ProducerCallback;
use serde::{Deserialize, Serialize};

use std::fmt::Debug;
// Reading files
use std::fs::File;
use std::hint::assert_unchecked;
use std::io::{BufReader, Read, Result as IoResult};
use std::iter::Zip;
use std::marker::PhantomData;
use std::ops::Range;
use std::path::Path;

use rayon::iter::plumbing::{Consumer, Producer, UnindexedConsumer, bridge};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};

use crate::utils::{conditionally_densify, prefetch_read_slice};
use crate::{ComponentType, SpaceUsage, ValueType};

// Implementation of a (immutable) sparse dataset.
pub type SparseDataset<C, V> = SparseDatasetGeneric<C, V, Box<[usize]>, Box<[C]>, Box<[V]>>;

/// A mutable representation of a sparse dataset.
///
/// This struct provides functionality for manipulating a sparse dataset with mutable access.
/// It stores sparse vectors in the dataset, where each vector consists of components
/// and their corresponding values of type `V`. The type `V` must implement the `SpaceUsage` trait,
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
pub type SparseDatasetMut<C, V> = SparseDatasetGeneric<C, V, Vec<usize>, Vec<C>, Vec<V>>;

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetGeneric<C, V, O, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]>,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
{
    dim: usize,
    offsets: O,
    components: AC,
    values: AV,
    _phantom: PhantomData<(C, V)>,
}

impl<C, V, O, AC, AV> SparseDatasetGeneric<C, V, O, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]>,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
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
    #[inline]
    pub fn get(&self, id: usize) -> (&[C], &[V]) {
        let range = self.offset_range(id);
        // SAFETY: Valid ID checked by `offset_range`.
        unsafe {
            let v_components = self.components.as_ref().get_unchecked(range.clone());
            let v_values = &self.values.as_ref().get_unchecked(range);

            (v_components, v_values)
        }
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
    /// let (components, values) = dataset.get_with_offset(3, 2);
    /// assert_eq!(components, &[1, 3]);
    /// assert_eq!(values, &[4.0, 5.0]);
    /// ```
    #[inline]
    pub fn get_with_offset(&self, offset: usize, len: usize) -> (&[C], &[V]) {
        unsafe { assert_unchecked(self.components.as_ref().len() == self.values.as_ref().len()) };

        let v_components = &self.components.as_ref()[offset..offset + len];
        let v_values = &self.values.as_ref()[offset..offset + len];

        (v_components, v_values)
    }

    /// Returns the range of positions of the slice with the given `id`.
    ///
    /// ### Panics
    /// Panics if the `id` is out of range.
    #[inline]
    pub fn offset_range(&self, id: usize) -> Range<usize> {
        let offsets = self.offsets.as_ref();
        assert!(id < offsets.len() - 1, "{id} is out of range");

        // Safety: safe accesses due to the check above
        unsafe {
            Range {
                start: *offsets.get_unchecked(id),
                end: *offsets.get_unchecked(id + 1),
            }
        }
    }

    /// Converts the `offset` of a vector within the dataset to its id, i.e., the position
    /// of the vector within the dataset.
    ///
    /// # Panics
    /// Panics if the `offset` is not the first position of a vector in the dataset.
    #[inline]
    pub fn offset_to_id(&self, offset: usize) -> usize {
        self.offsets.as_ref().binary_search(&offset).unwrap()
    }

    /// Prefetches the components and values of specified vectors into the CPU cache.
    ///
    /// This method prefetches the components and values of the vectors specified by their indices
    /// into the CPU cache, which can improve performance by reducing cache misses during subsequent accesses.
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
    /// // Prefetch components and values of vectors 0 and 1
    /// dataset.prefetch_vecs(&[0, 1]);
    /// ```
    #[inline]
    pub fn prefetch_vecs(&self, vecs: &[usize]) {
        for &id in vecs.iter() {
            let (components, values) = self.get(id);

            prefetch_read_slice(components);
            prefetch_read_slice(values);
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
    /// // Prefetch components and values of the vector starting at index 1 with length 3
    /// dataset.prefetch_vec_with_offset(1, 3);
    /// ```
    #[inline]
    pub fn prefetch_vec_with_offset(&self, offset: usize, len: usize) {
        let (components, values) = self.get_with_offset(offset, len);

        prefetch_read_slice(components);
        prefetch_read_slice(values);
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
    /// let query_components = &[0, 2];
    /// let query_values = &[1.0, 1.0];
    /// let k = 2;
    ///
    /// let knn = dataset.search(query_components, query_values, k);
    ///
    /// assert_eq!(knn, vec![(4.0, 2), (3.0, 0)]);
    /// ```
    pub fn search(&self, q_components: &[C], q_values: &[f32], k: usize) -> Vec<(f32, usize)> {
        let dense_query = conditionally_densify(q_components, q_values, self.dim());

        self.iter()
            .map(|(v_components, v_values)| {
                C::compute_dot_product(
                    dense_query.as_deref(),
                    q_components,
                    q_values,
                    v_components,
                    v_values,
                )
            })
            .enumerate()
            .map(|(i, s)| (s, i))
            .k_largest_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap())
            .collect()
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
    pub fn iter<'a>(&'a self) -> SparseDatasetIter<'a, C, V> {
        SparseDatasetIter::new(self)
    }

    pub fn par_iter<'a>(&'a self) -> ParSparseDatasetIter<'a, C, V> {
        ParSparseDatasetIter::new(self)
    }

    /// Returns an iterator over the sparse vector with id `vec_id`.
    ///
    /// # Panics
    /// Panics if the `vec_id` is out of bounds.
    pub fn iter_vector<'a>(
        &'a self,
        vec_id: usize,
    ) -> Zip<std::slice::Iter<'a, C>, std::slice::Iter<'a, V>> {
        let (v_components, v_values) = self.get(vec_id);

        v_components.iter().zip(v_values)
    }

    /// Returns the number of vectors in the dataset.
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
    ///                ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.offsets.as_ref().len() - 1
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
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.dim(), 5); // Largest component ID is 4, so dim() returns 5
    /// ```
    pub fn dim(&self) -> usize {
        self.dim
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
    pub fn nnz(&self) -> usize {
        self.components.as_ref().len()
    }

    // TODO: Do instead `impl<...> From<SparseDatasetGeneric<...>> for SparseDatasetGeneric<...>`.
    // However, this requires https://github.com/rust-lang/rust/issues/37653
    /// Converts a `SparseDatasetGeneric<D, f32, ...>` into a `SparseDatasetGeneric<C, V, ...>`, where `V` is can be casted from a `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use half::f16;
    /// use seismic::{SparseDataset, SparseDatasetMut};
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into_iter().collect();
    /// let dataset_f16 = SparseDataset::<u16, f16>::from_dataset_f32(dataset);
    ///
    /// assert_eq!(dataset_f16.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    pub fn from_dataset_f32<D, P, AD, AU>(dataset: SparseDatasetGeneric<D, f32, P, AD, AU>) -> Self
    where
        D: ComponentType,
        O: From<P>,
        P: AsRef<[usize]>,
        AC: From<AD>,
        AD: AsRef<[D]>,
        AV: From<Vec<V>>,
        AU: AsRef<[f32]>,
    {
        Self {
            dim: dataset.dim,
            offsets: dataset.offsets.into(),
            components: dataset.components.into(),
            values: dataset
                .values
                .as_ref()
                .iter()
                .map(|&v| V::from_f32_saturating(v))
                .collect_vec()
                .into(),
            _phantom: PhantomData,
        }
    }
}

impl<C, V> SparseDatasetMut<C, V>
where
    C: ComponentType,
    V: ValueType,
{
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
    pub fn read_bin_file(fname: &str) -> IoResult<Self> {
        Self::read_bin_file_limit(fname, None)
    }

    pub fn read_bin_file_limit(fname: &str, limit: Option<usize>) -> IoResult<Self> {
        let path = Path::new(fname);
        let f = File::open(path)?;
        //let f_size = f.metadata().unwrap().len() as usize;

        let mut br = BufReader::new(f);

        let mut buffer_d = [0_u8; std::mem::size_of::<u32>()];
        let mut buffer = [0_u8; std::mem::size_of::<f32>()];

        br.read_exact(&mut buffer_d)?;
        let mut n_vecs = u32::from_le_bytes(buffer_d) as usize;

        if let Some(n) = limit {
            n_vecs = n.min(n_vecs);
        }

        let mut data = Self::default();
        for _ in 0..n_vecs {
            br.read_exact(&mut buffer_d)?;
            let n = u32::from_le_bytes(buffer_d) as usize;

            let mut components = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);

            for _ in 0..n {
                br.read_exact(&mut buffer_d)?;
                let c = C::from_u32(u32::from_le_bytes(buffer_d)).unwrap();
                //.try_into()
                //.expect("Failed to convert to type C (Component Type).");
                components.push(c);
            }
            for _ in 0..n {
                br.read_exact(&mut buffer)?;
                let v = V::from_f32_saturating(f32::from_le_bytes(buffer));
                values.push(v);
            }

            data.push(&components, &values);
        }

        Ok(data)
    }
}

impl<C, V, O, AC, AV> Default for SparseDatasetGeneric<C, V, O, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + From<[usize; 1]>,
    AC: AsRef<[C]> + Default,
    AV: AsRef<[V]> + Default,
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
            dim: Default::default(),
            offsets: [0].into(),
            components: Default::default(),
            values: Default::default(),
            _phantom: PhantomData,
        }
    }
}

impl<C, V> SparseDatasetMut<C, V>
where
    C: ComponentType,
    V: ValueType,
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
    /// A new instance of `SparseDatasetMut<V>`.
    pub fn new() -> Self {
        Self::default()
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
    pub fn push(&mut self, components: &[C], values: &[V]) {
        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );

        assert!(
            components.is_sorted(),
            "Components must be given in sorted order"
        );

        if let Some(last_component) = components.last().map(|l| l.as_())
            && last_component >= self.dim
        {
            self.dim = last_component + 1;
        }

        self.components.extend(components);
        self.values.extend(values);
        self.offsets.push(self.components.len());
    }

    pub fn push_iterator(&mut self, pairs: impl Iterator<Item = (C, V)>) {
        // It would be nice to do `(self.components, self.values).extend(...)`, but that would move out of self...
        let (components, values): (Vec<_>, Vec<_>) = pairs.unzip();
        assert!(
            components.iter().is_sorted(),
            "Components must be given in sorted order"
        );

        if let Some(last_component) = components.last().map(|l| l.as_())
            && last_component >= self.dim
        {
            self.dim = last_component + 1;
        }

        self.components.extend(components);
        self.values.extend(values);
        self.offsets.push(self.components.len());
    }
}

impl<C, V, AC, AV> FromIterator<(AC, AV)> for SparseDatasetMut<C, V>
where
    C: ComponentType,
    V: ValueType,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
{
    /// Constructs a `SparseDatasetMut<V>` from an iterator over pairs of references to `[u16]` and `[V]`.
    ///
    /// This function consumes the provided iterator and constructs a new `SparseDatasetMut<V>`.
    /// Each pair in the iterator represents a pair of vectors, where the first vector contains
    /// the components and the second vector contains their corresponding values.
    ///
    /// # Parameters
    ///
    /// * `iter`: An iterator over pairs of vectors `(AsRef<[C]>, AsRef<[V]>)`.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDatasetMut<C, V>` populated with the pairs from the iterator.
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
        I: IntoIterator<Item = (AC, AV)>,
    {
        let mut dataset = SparseDatasetMut::<C, V>::new();

        for (components, values) in iter {
            dataset.push(components.as_ref(), values.as_ref());
        }

        dataset
    }
}

// Unfortunately, Rust doesn't yet support specialization, meaning that we can't use From too generically (otherwise it fails due to reimplementing `From<T> for T`)

impl<C, V> From<SparseDatasetMut<C, V>> for SparseDataset<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    /// Converts a mutable sparse dataset into an immutable one.
    ///
    /// This function consumes the provided `SparseDatasetMut<C, V>` and produces
    /// a corresponding immutable `SparseDataset<C, V>` instance. The conversion is performed
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
    fn from(dataset: SparseDatasetMut<C, V>) -> Self {
        Self {
            dim: dataset.dim,
            offsets: dataset.offsets.into(),
            components: dataset.components.into(),
            values: dataset.values.into(),
            _phantom: PhantomData,
        }
    }
}

impl<C, V> From<SparseDataset<C, V>> for SparseDatasetMut<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    /// Converts an immutable sparse dataset into a mutable one.
    ///
    /// This function consumes the provided `SparseDataset<C, V>` and produces
    /// a corresponding mutable `SparseDatasetMut<C, V>` instance. The conversion is performed
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
    fn from(dataset: SparseDataset<C, V>) -> Self {
        Self {
            dim: dataset.dim,
            offsets: dataset.offsets.into(),
            components: dataset.components.into(),
            values: dataset.values.into(),
            _phantom: PhantomData,
        }
    }
}

/// A struct to iterate over a sparse dataset. It assumes the dataset can be represented as a pair of slices.
#[derive(Clone)]
pub struct SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    #[inline]
    fn new<O, AC, AV>(dataset: &'a SparseDatasetGeneric<C, V, O, AC, AV>) -> Self
    where
        O: AsRef<[usize]>,
        AC: AsRef<[C]>,
        AV: AsRef<[V]>,
    {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets.as_ref()[1..],
            components: dataset.components.as_ref(),
            values: dataset.values.as_ref(),
        }
    }
}

impl<'a, C, V> Iterator for SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = (&'a [C], &'a [V]);

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

/// A struct to iterate over a sparse dataset in parallel.
/// It assumes the dataset can be represented as a pair of slices.
#[derive(Clone)]
pub struct ParSparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> ParSparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    #[inline]
    fn new<O, AC, AV>(dataset: &'a SparseDatasetGeneric<C, V, O, AC, AV>) -> Self
    where
        O: AsRef<[usize]>,
        AC: AsRef<[C]>,
        AV: AsRef<[V]>,
    {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets.as_ref()[1..],
            components: dataset.components.as_ref(),
            values: dataset.values.as_ref(),
        }
    }
}

impl<'a, C, V> ParallelIterator for ParSparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = (&'a [C], &'a [V]);

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

impl<C, V> IndexedParallelIterator for ParSparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
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

impl<C, V> ExactSizeIterator for SparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<C, V> DoubleEndedIterator for SparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
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
struct SparseDatasetProducer<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> Producer for SparseDatasetProducer<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = (&'a [C], &'a [V]);
    type IntoIter = SparseDatasetIter<'a, C, V>;

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

impl<'a, C, V> From<ParSparseDatasetIter<'a, C, V>> for SparseDatasetProducer<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn from(other: ParSparseDatasetIter<'a, C, V>) -> Self {
        Self {
            last_offset: other.last_offset,
            offsets: other.offsets,
            components: other.components,
            values: other.values,
        }
    }
}

impl<C, V, O, AC, AV> SpaceUsage for SparseDatasetGeneric<C, V, O, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + SpaceUsage,
    AC: AsRef<[C]> + SpaceUsage,
    AV: AsRef<[V]> + SpaceUsage,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.dim.space_usage_byte()
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

        let dataset = SparseDataset::<u16, f32>::from(dataset);

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
