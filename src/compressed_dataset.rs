use std::hash::Hash;
use std::hint::assert_unchecked;
use std::ops::Range;

use co_sort::*;
use compressed_intvec::prelude::{UIntVec, VariableCodecSpec};
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use serde::{Deserialize, Serialize};

use crate::partitioned_dataset::utils::build_or_load_metis_params;
use crate::sparse_dataset::SparseDatasetGeneric;
use crate::{
    ComponentType, SparseDatasetTrait, ValueType, distances::dot_product_dense_sparse,
    utils::prefetch_read_slice,
};
use crate::{FromDatasetGenericF32, SpaceUsage};

use mem_dbg::{MemSize, SizeFlags};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetCompressed<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    dim: usize,
    offsets: Box<[usize]>,
    components: UIntVec<C>,
    values: Box<[V]>,
    component_mapping: Box<[C]>,
}

impl<C, V> SparseDatasetTrait for SparseDatasetCompressed<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Component = C;
    type Value = V;
    type PreparedQuery<D: ComponentType, W: ValueType> = Vec<W>;

    #[inline]
    fn get_with_offset_iter(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (Self::Component, Self::Value)> {
        unsafe { assert_unchecked(len > 0) };
        let mut reader = self.components.seq_reader();
        gen move {
            for i in offset..(offset + len) {
                unsafe { yield (reader.get_unchecked(i), *self.values.get_unchecked(i)) }
            }
        }
    }

    fn prepare_query<D: ComponentType, W: ValueType>(
        &self,
        query: impl Iterator<Item = (D, W)>,
    ) -> Self::PreparedQuery<D, W> {
        let mut vec = vec![W::zero(); self.dim];
        for (i, v) in query {
            let mapped_component = self.component_mapping[i.as_()];
            vec[mapped_component.as_()] = v;
        }
        vec
    }

    fn dot_product_from_offset<D: ComponentType, W: ValueType>(
        &self,
        prepared_query: &Self::PreparedQuery<D, W>,
        offset: usize,
        len: usize,
    ) -> f32 {
        let cv = self
            .get_with_offset_iter(offset, len)
            .scan(0, |acc, (c, v)| {
                *acc += c.as_();
                Some((*acc, v.to_f32().unwrap()))
            });
        dot_product_dense_sparse(prepared_query, cv)
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn offset_to_id(&self, offset: usize) -> usize {
        self.offsets.as_ref().binary_search(&offset).unwrap()
    }

    #[inline]
    fn offset_range(&self, id: usize) -> Range<usize> {
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
    /// use crate::seismic::SparseDatasetTrait;
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into();
    ///
    /// // Prefetch components and values of the vector starting at index 1 with length 3
    /// dataset.prefetch_with_offset(1, 3);
    /// ```
    #[inline]
    fn prefetch_with_offset(&self, offset: usize, len: usize) {
        // TODO: The crate has no way to get where a certain index is actually stored...
        // prefetch_read_slice(components);
        prefetch_read_slice(unsafe { self.values.get_unchecked(offset..(offset + len)) });
    }

    fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = impl Iterator<Item = (Self::Component, Self::Value)>> {
        self.offsets
            .array_windows()
            .map(|&[start, end]| self.get_with_offset_iter(start, end - start))
    }

    fn par_iter(
        &self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = (Self::Component, Self::Value)>>
    {
        // https://github.com/rayon-rs/rayon/pull/789
        self.offsets.par_windows(2).map(|wind| {
            let &[start, end] = wind else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            self.get_with_offset_iter(start, end - start)
        })
    }
}

impl<C, V> SparseDatasetCompressed<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    pub fn search<D: ComponentType, W: ValueType>(
        &self,
        query: impl Iterator<Item = (D, W)>,
        k: usize,
    ) -> Vec<(f32, usize)> {
        let prepared_query = self.prepare_query(query);

        self.offsets
            .as_ref()
            .array_windows()
            .map(|&[o1, o2]| {
                let len = o2 - o1;
                self.dot_product_from_offset::<D, W>(&prepared_query, o1, len)
            })
            .enumerate()
            .map(|(i, s)| (s, i))
            .k_largest_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap())
            .collect()
    }
}

impl<C, V, O, AC, AV> FromDatasetGenericF32<SparseDatasetGeneric<C, f32, O, AC, AV>>
    for SparseDatasetCompressed<C, V>
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + SpaceUsage + Into<Box<[usize]>> + Hash,
    AC: AsRef<[C]> + SpaceUsage + Into<Box<[C]>> + Hash,
    AV: AsRef<[f32]> + SpaceUsage + Into<Box<[f32]>>,
{
    fn from_dataset_f32(dataset: SparseDatasetGeneric<C, f32, O, AC, AV>) -> Self {
        let metis_params = build_or_load_metis_params(&dataset);

        // Use partitioning so that components that often appear together have a close id
        let mut partitions = metis_params.build_partitions::<32>().into_boxed_slice();

        let dim = dataset.dim();
        let (offsets, components, values) = dataset.destroy();
        let mut component_ids: Box<[C]> = (0..dim).map(|c| C::from_usize(c).unwrap()).collect();
        co_sort_stable![partitions, component_ids];
        let mut component_mapping = vec![C::zero(); dim].into_boxed_slice();
        for (i, c) in component_ids.into_iter().enumerate() {
            component_mapping[c.as_()] = C::from_usize(i).unwrap();
        }

        let mut components = components.into();
        let mut values: Box<_> = values
            .into()
            .into_iter()
            .map(V::from_f32_saturating)
            .collect();
        let offsets = offsets.into();

        for &[start, end] in offsets.array_windows() {
            let comps = unsafe { components.get_unchecked_mut(start..end) };
            let vals = unsafe { values.get_unchecked_mut(start..end) };
            for c in comps.iter_mut() {
                *c = component_mapping[c.as_()];
            }
            co_sort!(comps, vals);

            // For better compression, store the component differences
            for i in (1..comps.len()).rev() {
                comps[i] = comps[i] - comps[i - 1];
            }
            // TODO: Completely adjust dot_product
        }

        // TODO: do we need a generic for the compressor type?
        Self {
            dim,
            offsets,
            components: UIntVec::<C>::builder()
                .k(64) // Sample every 2nd element
                .codec(VariableCodecSpec::Zeta { k: None })
                .build(components.as_ref())
                .unwrap(),
            values,
            component_mapping,
        }
    }
}

impl<C, V> SparseDatasetCompressed<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    pub fn space_usage_byte_components(&self) -> usize {
        self.components.mem_size(SizeFlags::CAPACITY)
    }

    pub fn from_dataset_f32_with_codec<O, AC, AV>(
        dataset: SparseDatasetGeneric<C, f32, O, AC, AV>,
        codec: VariableCodecSpec,
        permutation: Option<Box<[C]>>,
    ) -> Self
    where
        O: AsRef<[usize]> + SpaceUsage + Into<Box<[usize]>> + Hash,
        AC: AsRef<[C]> + SpaceUsage + Into<Box<[C]>> + Hash,
        AV: AsRef<[f32]> + SpaceUsage + Into<Box<[f32]>>,
    {
        let dim = dataset.dim();
        let permutation = permutation.unwrap_or_else(|| {
            println!("No permutation provided, computing one metis...");
            let metis_params = build_or_load_metis_params(&dataset);

            // Use partitioning so that components that often appear together have a close id
            let mut partitions = metis_params.build_partitions::<32>().into_boxed_slice();

            let mut component_ids: Box<[C]> = (0..dim).map(|c| C::from_usize(c).unwrap()).collect();
            co_sort_stable![partitions, component_ids];

            let mut component_mapping = vec![C::zero(); dim].into_boxed_slice();
            for (i, c) in component_ids.into_iter().enumerate() {
                component_mapping[c.as_()] = C::from_usize(i).unwrap();
            }
            component_mapping
        });

        let (offsets, components, values) = dataset.destroy();

        let mut components = components.into();
        let mut values: Box<_> = values
            .into()
            .into_iter()
            .map(V::from_f32_saturating)
            .collect();
        let offsets = offsets.into();

        for &[start, end] in offsets.array_windows() {
            let comps = unsafe { components.get_unchecked_mut(start..end) };
            let vals = unsafe { values.get_unchecked_mut(start..end) };
            for c in comps.iter_mut() {
                *c = permutation[c.as_()];
            }
            co_sort!(comps, vals);

            // For better compression, store the component differences
            for i in (1..comps.len()).rev() {
                comps[i] = comps[i] - comps[i - 1];
            }
        }
        let k_sampling = 128;
        println!("Using k={k_sampling} for sampling in compressed intvec");
        Self {
            dim,
            offsets,
            components: UIntVec::<C>::builder()
                .k(k_sampling)
                .codec(codec)
                .build(components.as_ref())
                .unwrap(),
            values,
            component_mapping: permutation,
        }
    }
}

impl<C, V> SpaceUsage for SparseDatasetCompressed<C, V>
where
    C: ComponentType,
    V: ValueType,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.dim.space_usage_byte()
            + self.offsets.space_usage_byte()
            //+ size_of_val(self.components.as_limbs())
            + self.components.mem_size(SizeFlags::CAPACITY)
            + self.values.space_usage_byte()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedU16Q;
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

        let compressed_dataset =
            SparseDatasetCompressed::<u16, FixedU16Q>::from_dataset_f32(dataset);

        let query = [(0_usize, 1.0), (1, 2.0)];
        let prepared_query = compressed_dataset.prepare_query(query.into_iter());

        // Gotta type the generic parameters because of https://github.com/rust-lang/rust/issues/144858
        assert_eq!(
            compressed_dataset.dot_product_from_id::<u16, _>(&prepared_query, 0),
            1.5
        );
        assert_eq!(
            compressed_dataset.dot_product_from_id::<u16, _>(&prepared_query, 1),
            1.0
        );
        assert_eq!(
            compressed_dataset.dot_product_from_id::<u16, _>(&prepared_query, 2),
            4.0
        );
    }
}
