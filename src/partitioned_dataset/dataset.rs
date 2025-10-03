use std::{
    hash::Hash,
    hint::{assert_unchecked, black_box},
    marker::PhantomData,
};

use bytemuck::{Pod, try_cast_slice};
use num_traits::{FromPrimitive, One, PrimInt, ToBytes, ToPrimitive};
use rayon::prelude::*;
use rusty_perm::*;
use serde::{Deserialize, Serialize};

use crate::{
    ComponentType, FromDatasetGenericF32, SpaceUsage, SparseDatasetTrait, ValueType,
    distances::dot_product_dense_sparse,
    partitioned_dataset::{fitting_integer::*, utils::*},
    sparse_dataset::SparseDatasetGeneric,
    utils::{build_or_load_metis_params, prefetch_read_slice},
};

trait Offset = PrimInt + FromPrimitive + ToPrimitive + Pod;

/// A view of a slice of `SparseDatasetPartitioned`, representing the document.
///
/// See its documentation for how documents are structured.
struct PostingView<'a, const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize, V, O>
where
    (): Fit<N_PARTITIONS> + Fit<N_COMPONENT_BITS>,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    O: Offset,
    V: ValueType,
{
    active_partitions: FittingArray<N_PARTITIONS>,
    active_partition_offsets: &'a [O],
    // This is technically an improper use of FittingInteger, since it caps to a `usize`...
    // But why would you use more than 64 bits for component, even normally?
    components_raw: &'a [FittingInteger<N_COMPONENT_BITS>],
    values: &'a [V],
}

impl<'a, const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize, V, O>
    PostingView<'a, N_PARTITIONS, N_COMPONENT_BITS, V, O>
where
    (): Fit<N_PARTITIONS>
        + Fit<N_COMPONENT_BITS>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>,
    [(); size_of::<FittingInteger<N_COMPONENT_BITS>>()]: Sized,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    O: Offset,
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

            let components_start =
                offsets_end.next_multiple_of(align_of::<FittingInteger<N_COMPONENT_BITS>>());
            let components_end = components_start
                + n_components_values * size_of::<FittingInteger<N_COMPONENT_BITS>>();
            let components_raw =
                try_cast_slice(slice.get_unchecked(components_start..components_end))
                    .unwrap_unchecked();

            let values_start = components_end.next_multiple_of(align_of::<V>());
            let values_end = values_start + n_components_values * size_of::<V>();
            let values =
                try_cast_slice(slice.get_unchecked(values_start..values_end)).unwrap_unchecked();

            Self {
                active_partitions,
                active_partition_offsets,
                components_raw,
                values,
            }
        }
    }

    /// Given a document, convert the document in a format that can be read by `Self::from_unchecked`.
    ///
    /// It asks for an existing `&mut Vec` to `extend` in order to avoid making unnecessary allocations.
    fn push_posting(
        vec: &mut Vec<u8>,
        converted_components: &mut [FittingInteger<
            { N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS },
        >],
        values: &mut [V],
    ) {
        let permutation = PermD::from_sort(&*converted_components);
        permutation.apply(values).unwrap();
        permutation.apply(converted_components).unwrap();

        let partitions_len: [_; N_PARTITIONS] =
            partitions_len_array(converted_components.iter(), |c| {
                c.unsigned_shr(N_COMPONENT_BITS as u32).as_()
            });

        let components_raw_iter = converted_components.iter().flat_map(|c| {
            let one = FittingInteger::<
                { N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS },
            >::one();
            let component_mask = one.unsigned_shl(N_COMPONENT_BITS as u32) - one;

            // Convert the components into bytes of `FittingInteger<N_COMPONENT_BITS>`
            let component_bits: FittingInteger<N_COMPONENT_BITS> =
                primitive_cast(*c & component_mask);
            // This sucks, complain to `num-traits`: https://github.com/rust-num/num-traits/pull/294
            *component_bits
                .to_ne_bytes()
                .as_ref()
                .as_array::<{ size_of::<FittingInteger<N_COMPONENT_BITS>>() }>()
                .unwrap()
        });

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
            vec.resize(
                vec.len()
                    .next_multiple_of(align_of::<FittingInteger<N_COMPONENT_BITS>>()),
                0,
            );
            vec.extend(components_raw_iter);
            vec.resize(vec.len().next_multiple_of(align_of::<V>()), 0);
            vec.extend_from_slice(try_cast_slice(values).unwrap_unchecked());
            vec.resize(vec.len().next_multiple_of(size_of::<usize>()), 0);
        }
    }

    fn active_partitions(&self) -> FittingArray<N_PARTITIONS> {
        unsafe {
            assert_unchecked(count_ones_array(&self.active_partitions) > 0);
        }
        self.active_partitions
    }

    unsafe fn components_values_nth_offset_raw(
        &self,
        n: usize,
    ) -> (&'a [FittingInteger<N_COMPONENT_BITS>], &'a [V]) {
        unsafe {
            let start = *self.active_partition_offsets.get_unchecked(n);
            let end = *self.active_partition_offsets.get_unchecked(n + 1);
            assert_unchecked(start < end);
            let components = self
                .components_raw
                .get_unchecked(start.to_usize().unwrap()..end.to_usize().unwrap());
            let values = self
                .values
                .get_unchecked(start.to_usize().unwrap()..end.to_usize().unwrap());
            (components, values)
        }
    }

    /// Iterate all components and values of the n-th active partition of the document. The partition part of the component is included.
    unsafe fn components_values_nth_offset_iter(
        &self,
        active_partition: usize,
        partition: usize,
    ) -> impl ExactSizeIterator<
        Item = (
            FittingInteger<
                { N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS },
            >,
            V,
        ),
    > + 'a {
        let (components, values) =
            unsafe { self.components_values_nth_offset_raw(active_partition) };
        let partition_shled = partition << N_COMPONENT_BITS;
        components.iter().zip(values).map(move |(&c, &v)| {
            let component = primitive_cast(c.as_() | partition_shled);
            // The reason for the `black_box` is to prevent the compiler from unrolling the loop in an attempt to be more efficient with many elements.
            // Because the number of elements is usually very small, the unrolling only bloats the code size and makes performance worse.
            black_box(());

            (component, v)
        })
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
                    // Without doing this, the result of `partition << N_COMPONENT_BITS` will be `or`ed every time, instead of just here.
                    let query = unsafe { query.get_unchecked((partition << N_COMPONENT_BITS)..) };

                    let n_skipped_offsets =
                        (self_active >> (n_bits - scroll)).count_ones() as usize;
                    current_offset = current_offset.wrapping_add(n_skipped_offsets);

                    let (components_raw, values) =
                        unsafe { self.components_values_nth_offset_raw(current_offset) };
                    let iter = components_raw.iter().zip(values.iter()).map(|(c, v)| {
                        // Same reason as `components_values_nth_offset_iter` above for `black_box`
                        black_box(());
                        (c.as_(), v.to_f32().unwrap())
                    });
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

    /// Iterate all components and values of the document, with the partition part of the component included.
    fn get_all_iter(
        self,
    ) -> impl Iterator<
        Item = (
            FittingInteger<
                { N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS },
            >,
            V,
        ),
    > {
        gen move {
            let n_bits = size_of::<FittingInteger<N_PARTITIONS>>() * 8;
            let mut active_partition_offset = 0;
            for (i, mut self_active) in self.active_partitions().into_iter().enumerate() {
                let mut partition = usize::MAX.wrapping_add(i * n_bits); // -1
                while !self_active.is_zero() {
                    let scroll = self_active.leading_zeros() as usize + 1;
                    partition = partition.wrapping_add(scroll);
                    self_active = self_active << scroll;

                    for cv in unsafe {
                        self.components_values_nth_offset_iter(active_partition_offset, partition)
                    } {
                        // Note: this returns the *converted* component, not the original
                        yield cv;
                    }
                    active_partition_offset += 1;
                }
            }
        }
    }
}

// Offsets take a big chunk of space, meaning that having them as small as possible is important.
// Thus, each posting's uses a different offset type depending on how many components/values in it.
// This macro helps choosing the correct Offset type given a posting's slice.
macro_rules! with_posting_view {
    ($view:expr, |$posting_view:ident| $body:block) => {
        unsafe {
            let view_u8 = try_cast_slice::<_, u8>($view).unwrap_unchecked();
            // This address is for getting the "second offset"
            // If the "second offset" is 0, it's because it's actually an u16
            if *view_u8.get_unchecked(size_of::<FittingArray<N_PARTITIONS>>() + 1) > 0 {
                let $posting_view =
                    PostingView::<N_PARTITIONS, N_COMPONENT_BITS, V, u8>::from_unchecked($view);
                $body
            } else {
                let $posting_view =
                    PostingView::<N_PARTITIONS, N_COMPONENT_BITS, V, u16>::from_unchecked($view);
                $body
            }
        }
    };
}

/// A dataset where in each document, components (and their respective values) are divided by the partition,
/// in order to require less information to represent.
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
pub struct SparseDatasetPartitioned<const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize, V>
where
    (): Fit<N_PARTITIONS>
        + Fit<N_COMPONENT_BITS>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>,
    [(); size_of::<FittingInteger<N_COMPONENT_BITS>>()]: Sized,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    V: ValueType,
{
    dim: usize,
    inflated_dim: usize,
    nnz: usize,
    offsets: Box<[usize]>,
    postings: Box<[usize]>, // usize to force alignment for each posting
    component_mapping: Box<
        [FittingInteger<
            { N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS },
        >],
    >,
    phantom_data: PhantomData<V>,
}

impl<const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize, V> SparseDatasetTrait
    for SparseDatasetPartitioned<N_PARTITIONS, N_COMPONENT_BITS, V>
where
    (): Fit<N_PARTITIONS>
        + Fit<N_COMPONENT_BITS>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>,
    [(); size_of::<FittingInteger<N_COMPONENT_BITS>>()]: Sized,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    V: ValueType,
{
    type Component =
        FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>;
    type Value = V;

    type PreparedQuery<D: ComponentType, U: ValueType> = (FittingArray<N_PARTITIONS>, Vec<U>);

    fn get_with_offset_iter(
        &self,
        offset: usize,
        len: usize,
    ) -> Box<dyn Iterator<Item = (Self::Component, Self::Value)> + Send + '_> {
        let view = unsafe { self.postings.get_unchecked(offset..offset + len) };
        with_posting_view!(view, |posting_view| {
            Box::new(posting_view.get_all_iter())
        })
    }

    fn prepare_query<D: ComponentType, U: ValueType>(
        &self,
        query: impl Iterator<Item = (D, U)>,
    ) -> Self::PreparedQuery<D, U> {
        let mut query_vec = vec![U::zero(); self.inflated_dim];

        let partitions: [_; N_PARTITIONS] = partitions_len_array(
            query.map(|(c, v)| {
                let mapped = unsafe { self.component_mapping.get(c.as_()).unwrap_unchecked() };
                query_vec[mapped.as_()] = v;
                (mapped, v)
            }),
            |(c, _)| c.unsigned_shr(N_COMPONENT_BITS as u32).to_usize().unwrap(),
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
        let view = unsafe { self.postings.get_unchecked(offset..offset + len) };
        with_posting_view!(view, |posting_view| {
            posting_view.dot_product(prepared_query.0, &prepared_query.1)
        })
    }

    fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn nnz(&self) -> usize {
        self.nnz
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

impl<const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize, C, V, O, AC, AV>
    FromDatasetGenericF32<SparseDatasetGeneric<C, f32, O, AC, AV>>
    for SparseDatasetPartitioned<N_PARTITIONS, N_COMPONENT_BITS, V>
where
    (): Fit<N_PARTITIONS>
        + Fit<N_COMPONENT_BITS>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>,
    [(); size_of::<FittingInteger<N_COMPONENT_BITS>>()]: Sized,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + SpaceUsage + IntoIterator<Item = usize> + Hash,
    AC: AsRef<[C]> + SpaceUsage + IntoIterator<Item = C> + Hash,
    AV: AsRef<[f32]> + SpaceUsage + IntoIterator<Item = f32>,
{
    fn from_dataset_f32(dataset: SparseDatasetGeneric<C, f32, O, AC, AV>) -> Self {
        let dim = dataset.dim();
        let nnz = dataset.nnz();

        let metis_params = build_or_load_metis_params(&dataset);

        // Build the partitioned dataset from the adjacency matrix
        let partitions: Box<_> = metis_params
            .build_partitions(N_PARTITIONS as i32)
            .into_iter()
            .map(|p| {
                FittingInteger::<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>::from_i32(p)
                    .unwrap()
            })
            .collect();

        let component_mapping = map_components::<N_PARTITIONS, N_COMPONENT_BITS>(&partitions);

        let mut postings_u8 = Vec::new();

        let (orig_offsets, orig_components, orig_values) = dataset.destroy();

        // Does not reallocate if `C` and `FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }`
        // are of the same size.
        let mut converted_components: Vec<_> = orig_components
            .into_iter()
            .map(|c| unsafe { *component_mapping.get_unchecked(c.as_()) })
            .collect();
        let inflated_dim = converted_components.iter().max().unwrap().as_() + 1;

        let mut values: Vec<_> = orig_values
            .into_iter()
            .map(V::from_f32_saturating)
            .collect();

        let offsets = std::iter::once(0)
            .chain(orig_offsets.into_iter().map_windows(|&[start, end]| {
                let c = unsafe { converted_components.get_unchecked_mut(start..end) };
                let v = unsafe { values.get_unchecked_mut(start..end) };

                if c.len() < 256 {
                    PostingView::<N_PARTITIONS, N_COMPONENT_BITS, V, u8>::push_posting(
                        &mut postings_u8,
                        c,
                        v,
                    );
                } else {
                    PostingView::<N_PARTITIONS, N_COMPONENT_BITS, V, u16>::push_posting(
                        &mut postings_u8,
                        c,
                        v,
                    );
                }
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
            inflated_dim,
            nnz,
            offsets,
            postings,
            component_mapping,
            phantom_data: PhantomData,
        }
    }
}

impl<const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize, V> SpaceUsage
    for SparseDatasetPartitioned<N_PARTITIONS, N_COMPONENT_BITS, V>
where
    (): Fit<N_PARTITIONS>
        + Fit<N_COMPONENT_BITS>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>,
    [(); size_of::<FittingInteger<N_COMPONENT_BITS>>()]: Sized,
    [(); N_PARTITIONS.div_ceil(size_of::<FittingInteger<N_PARTITIONS>>() * 8)]: Sized,
    V: ValueType,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.dim.space_usage_byte()
            + self.nnz.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.postings.space_usage_byte()
            + self.component_mapping.space_usage_byte()
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
            SparseDatasetPartitioned::<4, 1, FixedU16Q>::from_dataset_f32(dataset);

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
            SparseDatasetPartitioned::<32, 4, FixedU16Q>::from_dataset_f32(dataset);

        for (i, (components, values)) in [(c0, v0), (c1, v1)].into_iter().enumerate() {
            assert_eq!(
                partitioned_dataset
                    .get_iter(i)
                    .map(|(c, v)| {
                        let original_c = partitioned_dataset
                            .component_mapping
                            .iter()
                            .position(|x| *x == c)
                            .unwrap() as u16;
                        (original_c, v)
                    })
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

        let dataset = crate::SparseDatasetMut::<u8, f32>::from_iter(
            data.into_iter()
                .map(|(c, v)| c.into_iter().zip(v).collect_vec()),
        );

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

        let inverted_index =
            InvertedIndex::<SparseDatasetPartitioned<4, 1, FixedU16Q>>::from_base_dataset(
                dataset, config,
            );

        let (query_components, query_values) = ([0, 1], [1.0, 2.0]);

        let result = inverted_index.search(&query_components, &query_values, 5, 5, 0.7, 0, true);

        assert_eq!(result, vec![(4.0, 2), (1.5, 0), (1.0, 1)])
    }
}
