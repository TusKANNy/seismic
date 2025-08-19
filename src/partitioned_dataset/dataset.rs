use std::{
    hash::{DefaultHasher, Hash, Hasher},
    hint::{assert_unchecked, black_box},
    marker::PhantomData,
    ops::Range,
};

use bytemuck::{Pod, try_cast_slice};
use num_traits::{FromPrimitive, PrimInt, ToPrimitive};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    ComponentType, FromDatasetGenericF32, SpaceUsage, SparseDatasetTrait, ValueType,
    distances::dot_product_dense_sparse,
    partitioned_dataset::{fitting_integer::*, utils::*},
    sparse_dataset::SparseDatasetGeneric,
    utils::{prefetch_read_slice, read_from_path, write_to_path},
};

trait Offset = PrimInt + FromPrimitive + ToPrimitive + Pod;

/// A view of a slice of `SparseDatasetPartitioned`, representing the document.
///
/// See its documentation for how documents are structured.
struct PostingView<'a, const N_PARTITIONS: usize, C, V, O>
where
    (): Fit<N_PARTITIONS>,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    O: Offset,
    C: ComponentType,
    V: ValueType,
{
    active_partitions: FittingArray<N_PARTITIONS>,
    active_partition_offsets: &'a [O],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, const N_PARTITIONS: usize, C, V, O> PostingView<'a, N_PARTITIONS, C, V, O>
where
    (): Fit<N_PARTITIONS>,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    O: Offset,
    C: ComponentType,
    V: ValueType,
{
    /// Convert a slice from `SparseDatasetPartitioned`.
    // Is there a crate for automating this?
    unsafe fn from_unchecked(slice: &'a [usize]) -> Self {
        unsafe {
            let slice = try_cast_slice::<usize, u8>(slice).unwrap_unchecked();

            // Safe alternative in the potential future: https://github.com/rust-num/num-traits/issues/358
            let active_partitions = *(slice.as_ptr() as *const FittingArray<N_PARTITIONS>);

            let n_partitions = count_ones_array(&active_partitions) as usize;
            let offsets_start =
                size_of::<FittingArray<N_PARTITIONS>>().next_multiple_of(align_of::<O>());
            let offsets_end = offsets_start + (n_partitions + 1) * size_of::<O>();
            let active_partition_offsets: &[O] =
                try_cast_slice(slice.get_unchecked(offsets_start..offsets_end)).unwrap_unchecked();
            let n_components_values = active_partition_offsets
                .last()
                .unwrap_unchecked()
                .to_usize()
                .unwrap();

            let components_start = offsets_end.next_multiple_of(align_of::<C>());
            let components_end = components_start + n_components_values * size_of::<C>();
            let components = try_cast_slice(slice.get_unchecked(components_start..components_end))
                .unwrap_unchecked();

            let values_start = components_end.next_multiple_of(align_of::<V>());
            let values_end = values_start + n_components_values * size_of::<V>();
            let values =
                try_cast_slice(slice.get_unchecked(values_start..values_end)).unwrap_unchecked();

            Self {
                active_partitions,
                active_partition_offsets,
                components,
                values,
            }
        }
    }

    /// Given a document, convert the document in a format that can be read by `Self::from_unchecked`.
    ///
    /// It asks for an existing `&mut Vec` to `extend` in order to avoid making unnecessary allocations.
    fn push_posting(
        vec: &mut Vec<u8>,
        posting: impl Iterator<Item = (C, V)>,
        partitions: &[FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>],
    ) where
        (): Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>,
    {
        let partitioning_function = |c| partitioning_function(c, partitions);
        let (components, values) = sort_by_partition(posting, partitioning_function);
        let partitions_len: [_; N_PARTITIONS] =
            partitions_len_array(components.iter().cloned(), partitioning_function);

        let active_partitions =
            gen_active_partitions::<N_PARTITIONS>(partitions_len.map(|s| s > 0));

        let offsets: Vec<_> = std::iter::once(O::zero())
            .chain(
                partitions_len
                    .iter()
                    .filter(|&&s| s > 0)
                    .scan(0, |acc, &l| {
                        *acc += l;
                        Some(unsafe { O::from_usize(*acc).unwrap_unchecked() })
                    }),
            )
            .collect();

        unsafe {
            vec.extend_from_slice(try_cast_slice(&active_partitions).unwrap_unchecked());
            vec.resize(vec.len().next_multiple_of(align_of::<O>()), 0);
            vec.extend_from_slice(try_cast_slice(offsets.as_slice()).unwrap_unchecked());
            vec.resize(vec.len().next_multiple_of(align_of::<C>()), 0);
            vec.extend_from_slice(try_cast_slice(components.as_slice()).unwrap_unchecked());
            vec.resize(vec.len().next_multiple_of(align_of::<V>()), 0);
            vec.extend_from_slice(try_cast_slice(values.as_slice()).unwrap_unchecked());
            vec.resize(vec.len().next_multiple_of(size_of::<usize>()), 0);
        }
    }

    fn active_partitions(&self) -> FittingArray<N_PARTITIONS> {
        unsafe {
            assert_unchecked(count_ones_array(&self.active_partitions) > 0);
        }
        self.active_partitions
    }

    unsafe fn component_values_nth_offset_raw(&self, n: usize) -> (&'a [C], &'a [V]) {
        unsafe {
            let start = *self.active_partition_offsets.get_unchecked(n);
            let end = *self.active_partition_offsets.get_unchecked(n + 1);
            assert_unchecked(start < end);
            let components = self
                .components
                .get_unchecked(start.to_usize().unwrap()..end.to_usize().unwrap());
            let values = self
                .values
                .get_unchecked(start.to_usize().unwrap()..end.to_usize().unwrap());
            (components, values)
        }
    }

    /// Iterate all components and values of the n-th active partition of the document.
    unsafe fn components_values_nth_offset_iter(
        &self,
        active_partition: usize,
        _partition: usize,
    ) -> impl ExactSizeIterator<Item = (C, V)> + 'a {
        let (components, values) =
            unsafe { self.component_values_nth_offset_raw(active_partition) };
        components
            .iter()
            .zip(values)
            // The reason for the `black_box` is to prevent the compiler from unrolling the loop in an attempt to be more efficient with many elements.
            // Because the number of elements is usually very small, the unrolling only bloats the code size and makes performance worse.
            .map(|(&c, &v)| (c, black_box(v)))
    }

    /// Perform the dot product of this document with a given query.
    // Instead of hardcoding `dot_product_dense_sparse`, it'd be better have as argument a generic function that takes an Iterator as a parameter.
    // However, doing `impl Fn(impl Iterator)` requires Higher-Rank Trait Bounds, which aren't in Rust yet.
    fn dot_product<T: ValueType>(&'a self, mask: FittingArray<N_PARTITIONS>, query: &[T]) -> f32 {
        let mut result: f32 = 0.0;

        let n_bits = size_of::<FittingInteger<N_PARTITIONS>>() * 8;

        let self_active_arr = self.active_partitions();
        let mut starting_offset = usize::MAX; // -1

        // The loop is unrolled
        for (i, (mut self_active, cur_mask)) in self_active_arr.into_iter().zip(mask).enumerate() {
            // Only iterate the partitions that are active for both the document and the query, in order to skip the components and values that aren't in active partitions
            let mut query_active = self_active & cur_mask;
            let mut partition = usize::MAX.wrapping_add(i * n_bits); // Starts from -1
            let mut current_offset = starting_offset;
            starting_offset = starting_offset.wrapping_add(self_active.count_ones() as usize); // Add the count of `1`s for the next iteration.

            // Do NOT do `if query_active.is_zero() {continue;}`, as that prevents the previous loop from unrolling.
            if !query_active.is_zero() {
                let mut n_active = query_active.count_ones();
                // Iterate the `1` bits
                loop {
                    let scroll = query_active.leading_zeros() as usize + 1;

                    partition = partition.wrapping_add(scroll);

                    let n_skipped_offsets =
                        (self_active >> (n_bits - scroll)).count_ones() as usize;
                    current_offset = current_offset.wrapping_add(n_skipped_offsets);
                    let iter = unsafe {
                        self.components_values_nth_offset_iter(current_offset, partition)
                            .map(|(c, v)| (c.as_(), v.to_f32().unwrap()))
                    };
                    result = result.algebraic_add(dot_product_dense_sparse(query, iter));

                    n_active -= 1;
                    if n_active == 0 {
                        break;
                    }
                    unsafe {
                        // On Rust, shifts are wrapping. The `assert` prevents the compiler from adding an `and` instruction.
                        // SAFETY: Equality can only happen with a singular 1 at the end.
                        // But because it has to be the only one, this loop must have already `break`ed.
                        assert_unchecked(scroll < n_bits);
                    }
                    self_active = self_active << scroll;
                    query_active = query_active << scroll;
                }
            }
        }

        result
    }

    /// Iterate all the components and values of the document.
    fn get_all_iter(self) -> impl ExactSizeIterator<Item = (C, V)> {
        self.components
            .iter()
            .zip(self.values)
            .map(|(&c, &v)| (c, v))
    }
}

/// A dataset where in each document, components (and their respective values) are divided by the partition,
/// in order to require less information to represent (not implemented yet).
///
/// The `postings` array stores the documents with a `usize` alignment (converted into a `u8` slice for actual use).
///
/// In this context, an "active partition" of a document is a partition such that there exists at least one component belonging to it in the document.
///
/// All fields of the document are aligned:
/// * `active_partitions`: A statically sized array of `usize`s, representing a bitset of the active partitions.
///   The size of the array depends on `N_PARTITIONS`. If the size is 1, the type will be the smallest possible unsigned integer that fits the bitfield.
/// * `active_partition_offsets`: A slice representing the starting and ending positions of the `components` and `values` for each active partition.
///   Its length is the number of `1`s in `active_partitions`, plus 1 (as it also contains `0` and the end).
/// * `components` and `values` are ordered such that each slice determined by `active_partition_offsets[i]..active_partition_offsets[i+1]` belongs to the same partition.
///   The length of both their arrays is determined by `active_partition_offsets.last()`.
#[derive(Serialize, Deserialize)]
pub struct SparseDatasetPartitioned<const N_PARTITIONS: usize, C, V>
where
    (): Fit<N_PARTITIONS> + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>,
    C: ComponentType,
    V: ValueType,
{
    dim: usize,
    nnz: usize,
    offsets: Box<[usize]>,
    postings: Box<[usize]>, // usize to force alignment for each posting
    partitions: Box<[FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>]>, // Could be u8 (why use more than 256 partitions?), but this is technically correct
    phantom_data: PhantomData<(C, V)>,
}

impl<const N_PARTITIONS: usize, C, V> SparseDatasetTrait
    for SparseDatasetPartitioned<N_PARTITIONS, C, V>
where
    (): Fit<N_PARTITIONS> + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    C: ComponentType,
    V: ValueType,
{
    type Component = C;
    type Value = V;

    type PreparedQuery<D: ComponentType, U: ValueType> = (FittingArray<N_PARTITIONS>, Vec<U>);

    fn get_with_offset_iter(
        &self,
        offset: usize,
        len: usize,
    ) -> impl ExactSizeIterator<Item = (Self::Component, Self::Value)> {
        let posting_view = unsafe {
            PostingView::<N_PARTITIONS, C, V, u16>::from_unchecked(
                self.postings.get_unchecked(offset..offset + len),
            )
        };
        posting_view.get_all_iter()
    }

    fn prepare_query<D: ComponentType, U: ValueType>(
        &self,
        query: impl Iterator<Item = (D, U)>,
    ) -> Self::PreparedQuery<D, U> {
        let mut query_vec = vec![U::zero(); self.dim()];

        let partitions: [_; N_PARTITIONS] = partitions_len_array(
            query.inspect(|&(c, v)| {
                query_vec[c.as_()] = v;
            }),
            |(d, _)| partitioning_function(d, self.partitions.as_ref()),
        );
        let active_partitions = gen_active_partitions(partitions.map(|l| l > 0));

        (active_partitions, query_vec)
    }

    fn dot_product_from_offset<D: ComponentType, U: ValueType>(
        &self,
        prepared_query: &Self::PreparedQuery<D, U>,
        offset: usize,
        len: usize,
    ) -> f32 {
        unsafe {
            PostingView::<N_PARTITIONS, C, V, u16>::from_unchecked(
                self.postings.get_unchecked(offset..offset + len),
            )
        }
        .dot_product(prepared_query.0, &prepared_query.1)
    }

    fn offset_to_id(&self, offset: usize) -> usize {
        self.offsets.as_ref().binary_search(&offset).unwrap()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn len(&self) -> usize {
        self.offsets.as_ref().len() - 1
    }

    fn offset_range(&self, id: usize) -> std::ops::Range<usize> {
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

    fn prefetch_with_offset(&self, offset: usize, len: usize) {
        let posting = unsafe { self.postings.get_unchecked(offset..offset + len) };
        prefetch_read_slice(posting);
    }

    fn iter(&self) -> impl Iterator<Item = impl Iterator<Item = (Self::Component, Self::Value)>> {
        self.offsets
            .array_windows()
            .map(|&[start, end]| self.get_with_offset_iter(start, end - start))
    }

    fn par_iter(
        &self,
    ) -> impl rayon::prelude::IndexedParallelIterator<
        Item = impl Iterator<Item = (Self::Component, Self::Value)> + Send,
    > {
        // https://github.com/rayon-rs/rayon/pull/789
        self.offsets.par_windows(2).map(|wind| {
            let &[start, end] = wind else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            self.get_with_offset_iter(start, end - start)
        })
    }
}

impl<const N_PARTITIONS: usize, C, V, O, AC, AV>
    FromDatasetGenericF32<SparseDatasetGeneric<C, f32, O, AC, AV>>
    for SparseDatasetPartitioned<N_PARTITIONS, C, V>
where
    (): Fit<N_PARTITIONS> + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + SpaceUsage + Hash,
    AC: AsRef<[C]> + SpaceUsage + Hash,
    AV: AsRef<[f32]> + SpaceUsage,
{
    fn from_dataset_f32(dataset: SparseDatasetGeneric<C, f32, O, AC, AV>) -> Self {
        let dim = dataset.dim();
        let nnz = dataset.nnz();

        // Load the dataset's adjacency matrix
        // Hash the dataset's components and offsets so that its adjacency can be cached (as it's a very long operation)
        let mut s = DefaultHasher::new();
        dataset.components().hash(&mut s);
        dataset.offsets().hash(&mut s);
        let hash = s.finish();
        let filename = format!("cached_adjacency_{}", hash);
        let metis_params = if !std::fs::exists(filename.as_str()).is_ok_and(|b| b) {
            println!("Adjacency matrix not cached. Creating.");
            let params = dataset.adjacency_matrix_metis();

            println!("Saving ... {}", filename);
            write_to_path(&params, filename.as_str()).unwrap();

            params
        } else {
            println!("Loading adjacency matrix {}.", filename.as_str());
            read_from_path(filename.as_str()).unwrap()
        };

        // Build the partitioned dataset from the adjacency matrix
        let partitions = metis_params
            .build_partitions::<N_PARTITIONS>()
            .into_boxed_slice();

        let mut postings_u8 = Vec::new();

        let offsets = std::iter::once(0)
            .chain(dataset.iter().map(|p| {
                let p = p.map(|(c, v)| (c, V::from_f32_saturating(v)));
                PostingView::<N_PARTITIONS, C, V, u16>::push_posting(
                    &mut postings_u8,
                    p,
                    partitions.as_ref(),
                );
                postings_u8.len() / size_of::<usize>()
            }))
            .collect();

        // I hate that I have to reallocate to guarantee alignment, Rust is currently really bad at these kinds of things.
        let mut postings = Vec::with_capacity(postings_u8.len() / size_of::<usize>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                postings_u8.as_mut_ptr(),
                postings.spare_capacity_mut().as_mut_ptr() as *mut u8,
                postings_u8.len(),
            );
            postings.set_len(postings.capacity());
        };
        let postings = postings.into_boxed_slice();

        Self {
            dim,
            nnz,
            offsets,
            postings,
            partitions,
            phantom_data: PhantomData,
        }
    }
}

impl<const N_PARTITIONS: usize, C, V> SpaceUsage for SparseDatasetPartitioned<N_PARTITIONS, C, V>
where
    (): Fit<N_PARTITIONS> + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>,
    C: ComponentType,
    V: ValueType,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.dim.space_usage_byte()
            + self.nnz.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.postings.space_usage_byte()
            + self.partitions.space_usage_byte()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedU16Q;
    use crate::inverted_index::*;
    use itertools::Itertools;

    #[test]
    fn conversion_and_dot_product() {
        let data = [
            (vec![0, 2, 4], vec![1.5, 2.0, 2.5]),
            (vec![1, 3], vec![0.5, 1.0]),
            (vec![0, 1, 2, 3], vec![1.0, 1.5, 2.0, 2.5]),
        ];

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.into_iter()
                .map(|(c, v)| c.into_iter().zip(v).collect_vec()),
        );

        let partitioned_dataset =
            SparseDatasetPartitioned::<4, u16, FixedU16Q>::from_dataset_f32(dataset);

        let query = [(0_usize, 1.0), (1, 2.0)];
        let prepared_query = partitioned_dataset.prepare_query(query.into_iter());

        // Gotta type the generic parameters because of https://github.com/rust-lang/rust/issues/144858
        assert_eq!(
            partitioned_dataset.dot_product_from_id::<u16, _>(&prepared_query, 0),
            1.5
        );
        assert_eq!(
            partitioned_dataset.dot_product_from_id::<u16, _>(&prepared_query, 1),
            1.0
        );
        assert_eq!(
            partitioned_dataset.dot_product_from_id::<u16, _>(&prepared_query, 2),
            4.0
        );
    }

    #[test]
    fn test_iteration() {
        let data = [
            (vec![0, 2, 4], vec![1.5, 2.0, 2.5]),
            ((0..260).collect(), vec![1.0; 260]),
        ];
        let [(c0, v0), (c1, v1)] = data.clone();

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.into_iter()
                .map(|(c, v)| c.into_iter().zip(v).collect_vec()),
        );

        let partitioned_dataset =
            SparseDatasetPartitioned::<32, u16, FixedU16Q>::from_dataset_f32(dataset);

        for (i, (components, values)) in [(c0, v0), (c1, v1)].into_iter().enumerate() {
            assert_eq!(
                partitioned_dataset
                    .get_iter(i)
                    .sorted_unstable_by_key(|(c, _)| *c)
                    .collect_vec(),
                (components.into_iter().zip(
                    values
                        .into_iter()
                        .map(|n| FixedU16Q::saturating_from_num(n))
                ))
                .collect_vec()
            );
        }
    }

    #[test]
    fn test_inverted_index() {
        let data = [
            (vec![0, 2, 4], vec![1.5, 2.0, 2.5]),
            (vec![1, 3], vec![0.5, 1.0]),
            (vec![0, 1, 2, 3], vec![1.0, 1.5, 2.0, 2.5]),
        ];

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.into_iter()
                .map(|(c, v)| c.into_iter().zip(v).collect_vec()),
        );

        let partitioned_dataset =
            SparseDatasetPartitioned::<4, u16, FixedU16Q>::from_dataset_f32(dataset);

        let my_clustering_algorithm = ClusteringAlgorithm::RandomKmeans {};

        let config = Configuration::default()
            .pruning_strategy(PruningStrategy::FixedSize { n_postings: 10 })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction: 0.1,
                min_cluster_size: 2,
                clustering_algorithm: my_clustering_algorithm,
            })
            .summarization_strategy(SummarizationStrategy::EnergyPreserving {
                summary_energy: 0.4,
            });

        println!("\nBuilding the index...");
        println!("{:?}", config);

        let inverted_index = InvertedIndex::build(partitioned_dataset, config);

        let (query_components, query_values) = ([0, 1], [1.0, 2.0]);

        let result = inverted_index.search(&query_components, &query_values, 5, 5, 0.7, 0, true);

        assert_eq!(result, vec![(4.0, 2), (1.5, 0), (1.0, 1)])
    }
}
