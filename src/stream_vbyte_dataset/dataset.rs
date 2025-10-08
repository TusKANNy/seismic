use std::hash::Hash;

use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    ComponentType, FixedU8Q, FromDatasetGenericF32, SimdyValueType, SpaceUsage, SparseDatasetTrait,
    ValueType,
    sparse_dataset::SparseDatasetGeneric,
    stream_vbyte_dataset::stream_vbyte::{StreamVbyte, ToF32Simd},
    utils::{permute_or_load_with_graph_bisection, prefetch_read},
};

#[derive(Serialize, Deserialize)]
pub struct SparseDatasetStreamVbyte<T>
where
    T: ValueType + fixed::traits::Fixed,
    T::Bits: SimdyValueType + Send + Sync, // Fixed asks for this
{
    dim: usize,
    nnz: usize,
    offsets: Box<[usize]>,
    posting_lengths: Box<[u16]>,
    postings: Box<[usize]>, // usize to force alignment for each posting
    component_mapping: Box<[u16]>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> SparseDatasetStreamVbyte<T>
where
    T: ValueType + fixed::traits::Fixed,
    T::Bits: SimdyValueType + Send + Sync + ToF32Simd, // Fixed asks for this
{
    unsafe fn get_stream_vbyte_from_offset(
        &'_ self,
        offset: usize,
        len: usize,
    ) -> StreamVbyte<'_, T::Bits> {
        unsafe {
            let view = self.postings.get_unchecked(offset..);
            StreamVbyte::from_unchecked(view, len)
        }
    }
}

impl<T> SparseDatasetTrait for SparseDatasetStreamVbyte<T>
where
    T: ValueType + fixed::traits::Fixed,
    T::Bits: SimdyValueType + Send + Sync + ToF32Simd, // Fixed asks for this
{
    type Component = u16;
    type Value = T;

    type PreparedQuery<D: ComponentType, U: ValueType> = Vec<f32>;

    fn get_with_offset_iter(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (Self::Component, Self::Value)> + '_ {
        let stream_view: StreamVbyte<'_, T::Bits> =
            unsafe { self.get_stream_vbyte_from_offset(offset, len) };
        stream_view.iter().map(|(c, v)| (c, T::from_bits(v)))
    }

    fn prepare_query<D: ComponentType, U: ValueType>(
        &self,
        query: impl Iterator<Item = (D, U)>,
    ) -> Self::PreparedQuery<D, U> {
        let mut vec = vec![0.0; self.dim];
        for (c, v) in query {
            let mapped_component = self.component_mapping[c.as_()];
            vec[mapped_component as usize] = v.to_f32().unwrap();
        }

        vec
    }

    fn dot_product_from_offset<D: ComponentType, U: ValueType>(
        &self,
        prepared_query: &Self::PreparedQuery<D, U>,
        offset: usize,
        len: usize,
    ) -> f32 {
        let stream_view = unsafe { self.get_stream_vbyte_from_offset(offset, len) };
        stream_view.dot_product(prepared_query)
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

    fn offset_range(&self, id: usize) -> std::ops::Range<usize> {
        let offset = *unsafe { self.offsets.get_unchecked(id) };
        // TODO: This is horrible, but it's the best way I can think of tto only have one value needed for the dot product
        // Maybe the function should be renamed, almost everything uses it for the "length" after all.
        offset..offset + *unsafe { self.posting_lengths.get_unchecked(id) } as usize
    }

    fn prefetch_with_offset(&self, offset: usize, _len: usize) {
        let posting = unsafe { self.postings.get_unchecked(offset) };
        prefetch_read(posting);
    }

    fn iter(&self) -> impl Iterator<Item = impl Iterator<Item = (Self::Component, Self::Value)>> {
        self.offsets
            .iter()
            .zip(self.posting_lengths.iter())
            .map(|(&offset, &len)| self.get_with_offset_iter(offset, len as usize))
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

impl<C, T, O, AC, AV> FromDatasetGenericF32<SparseDatasetGeneric<C, f32, O, AC, AV>>
    for SparseDatasetStreamVbyte<T>
where
    T: ValueType + fixed::traits::Fixed,
    T::Bits: SimdyValueType + Send + Sync + ToF32Simd,
    C: ComponentType + DeserializeOwned + Serialize + Hash,
    O: AsRef<[usize]> + SpaceUsage + IntoIterator<Item = usize> + Hash,
    AC: AsRef<[C]> + SpaceUsage + IntoIterator<Item = C> + Hash,
    AV: AsRef<[f32]> + SpaceUsage + IntoIterator<Item = f32>,
{
    fn from_dataset_f32(dataset: SparseDatasetGeneric<C, f32, O, AC, AV>) -> Self {
        let dim = dataset.dim();
        let nnz = dataset.nnz();

        // TEMPORARY: Using graph bisection instead of METIS for testing
        // TODO: Restore METIS-based partitioning if needed
        //
        // Old METIS-based code (commented out temporarily):
        // let metis_params = build_or_load_metis_params(&dataset);
        // let partitions = metis_params.build_partitions(32).into_boxed_slice();
        // let permutation = PermD::from_sort(partitions.as_ref());
        // let component_mapping: Box<[u16]> =
        //     permutation.indices().iter().map(|&n| n as u16).collect();

        // Use graph bisection to compute a permutation that groups related components
        let perm = permute_or_load_with_graph_bisection(&dataset);

        // Create a simple mapping from the permutation
        let component_mapping: Box<[u16]> = perm.iter().map(|&p| p.as_() as u16).collect();

        let mut postings_u8 = Vec::new();

        let (orig_offsets, orig_components, orig_values) = dataset.destroy();

        let mut converted_components: Vec<_> = orig_components
            .into_iter()
            .map(|c| unsafe { *component_mapping.get_unchecked(c.as_()) })
            .collect();

        let mut values: Vec<_> = orig_values
            .into_iter()
            .map(|v| FixedU8Q::from_f32_saturating(v).to_bits())
            .collect();

        let mut posting_lengths = Vec::with_capacity(nnz);

        let offsets = std::iter::once(0)
            .chain(orig_offsets.into_iter().map_windows(|&[start, end]| {
                let c = unsafe { converted_components.get_unchecked_mut(start..end) };
                let v = unsafe { values.get_unchecked_mut(start..end) };
                posting_lengths.push(c.len() as u16);

                StreamVbyte::<T::Bits>::push_posting(&mut postings_u8, c, v);
                postings_u8.len() / size_of::<usize>()
            }))
            .collect();
        let posting_lengths = posting_lengths.into_boxed_slice();

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
            component_mapping,
            posting_lengths,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SpaceUsage for SparseDatasetStreamVbyte<T>
where
    T: ValueType + fixed::traits::Fixed,
    T::Bits: SimdyValueType + Send + Sync,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.dim.space_usage_byte()
            + self.nnz.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.posting_lengths.space_usage_byte()
            + self.postings.space_usage_byte()
            + self.component_mapping.space_usage_byte()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverted_index::*;
    use itertools::Itertools;

    #[test]
    fn conversion_and_dot_product() {
        let data = [
            (vec![500, 502, 504], vec![1.5, 2.0, 2.5]),
            (vec![1, 3], vec![0.5, 1.0]),
            (vec![0, 1, 2, 3], vec![1.0, 1.5, 2.0, 2.5]),
        ];

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.into_iter()
                .map(|(c, v)| c.into_iter().zip(v).collect_vec()),
        );

        let streamed_dataset = SparseDatasetStreamVbyte::<FixedU8Q>::from_dataset_f32(dataset);

        let query_1 = [(0_usize, 1.0), (1, 2.0)];
        let prepared_query_1 = streamed_dataset.prepare_query(query_1.into_iter());

        let query_2 = [(500_usize, 1.0), (501, 2.0)];
        let prepared_query_2 = streamed_dataset.prepare_query(query_2.into_iter());

        // Gotta type the generic parameters because of https://github.com/rust-lang/rust/issues/144858
        assert_eq!(
            streamed_dataset.dot_product_from_id::<u16, f32>(&prepared_query_2, 0),
            1.5
        );
        assert_eq!(
            streamed_dataset.dot_product_from_id::<u16, f32>(&prepared_query_1, 1),
            1.0
        );
        assert_eq!(
            streamed_dataset.dot_product_from_id::<u16, f32>(&prepared_query_1, 2),
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

        let streamed_dataset = SparseDatasetStreamVbyte::from_dataset_f32(dataset);

        for (i, (components, values)) in [(c0, v0), (c1, v1)].into_iter().enumerate() {
            assert_eq!(
                streamed_dataset
                    .get_iter(i)
                    .map(|(c, v)| {
                        let original_c = streamed_dataset
                            .component_mapping
                            .iter()
                            .position(|x| *x == c)
                            .unwrap() as u16;
                        (original_c, v)
                    })
                    .sorted_unstable_by_key(|(c, _)| *c)
                    .collect_vec(),
                (components
                    .into_iter()
                    .zip(values.into_iter().map(|n| FixedU8Q::saturating_from_num(n))))
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
            InvertedIndex::<SparseDatasetStreamVbyte<FixedU8Q>>::from_base_dataset(dataset, config);

        let (query_components, query_values) = ([0, 1], [1.0, 2.0]);

        let result = inverted_index.search(&query_components, &query_values, 5, 5, 0.7, 0, true);

        assert_eq!(result, vec![(4.0, 2), (1.5, 0), (1.0, 1)])
    }
}
