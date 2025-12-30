use std::collections::HashSet;

use crate::QuantizedSummary;
use crate::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, SummarizationStrategy,
};
use crate::utils::{
    KHeap, do_random_kmeans_on_docids, do_random_kmeans_on_docids_ii_approx_dot_product,
    do_random_kmeans_on_docids_ii_dot_product,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use num_traits::ToPrimitive;
use vectorium::dataset::ScoredRangeDotProduct;
use vectorium::{
    ComponentType, Dataset, Distance, DotProduct, GrowableDataset, QueryEvaluator, SpaceUsage,
    SparseDataset, SparseDatasetGrowable, SparseVector1D, SparseVectorEncoder, Vector1D,
    VectorEncoder,
};

type ComponentFor<E> = <E as VectorEncoder>::OutputComponentType;
type ValueFor<E> = <E as VectorEncoder>::OutputValueType;
type QueryValueFor<E> = <E as VectorEncoder>::QueryValueType;

/// Instead of storing doc_ids we store their offsets in the forward_index and the lengths of the vectors
/// This allows us to save the random accesses that would be needed to access exactly these values from the
/// forward index. The values of each doc are packed into a single u64 in `packed_postings`.
/// We use 48 bits for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48.
/// We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash, Ord, PartialOrd)]
pub(crate) struct PackedPostingBlock {
    pub n: u64,
}

impl PackedPostingBlock {
    #[inline]
    pub fn pack(range: std::ops::Range<usize>) -> Self {
        let start = range.start as u64;
        assert!(
            start < (1u64 << 48),
            "range.start exceeds 48-bit packing limit"
        );
        let len = range.len();
        assert!(
            len <= u16::MAX as usize,
            "range length exceeds 16-bit packing limit"
        );
        Self {
            n: (start << 16) | (len as u64),
        }
    }

    #[inline]
    pub fn unpack(&self) -> std::ops::Range<usize> {
        let start = (self.n >> 16) as usize;
        let len = (self.n & (u16::MAX as u64)) as usize;
        start..(start + len)
    }
}

impl SpaceUsage for PackedPostingBlock {
    fn space_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PostingList<C: ComponentType> {
    pub(crate) packed_postings: Box<[PackedPostingBlock]>,
    pub(crate) block_offsets: Box<[usize]>,
    pub(crate) summaries: QuantizedSummary<C>,
}

impl<C: ComponentType + SpaceUsage> SpaceUsage for PostingList<C> {
    fn space_usage_bytes(&self) -> usize {
        SpaceUsage::space_usage_bytes(&self.packed_postings)
            + SpaceUsage::space_usage_bytes(&self.block_offsets)
            + self.summaries.space_usage_bytes()
    }
}

impl<C: ComponentType> PostingList<C> {
    pub(crate) fn get_all_doc_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::with_capacity(self.packed_postings.len());
        for pack in self.packed_postings.iter() {
            ranges.push(pack.unpack());
        }

        ranges
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search<S, E, QE, QC, QV>(
        &self,
        evaluator: &QE,
        query: &SparseVector1D<C, QueryValueFor<E>, QC, QV>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRangeDotProduct>,
        visited: &mut HashSet<usize>,
        forward_index: &S,
    ) where
        S: Dataset<E>,
        E: VectorEncoder<Distance = DotProduct>,
        QE: QueryEvaluator<E>,
        QC: AsRef<[C]>,
        QV: AsRef<[QueryValueFor<E>]>,
    {
        let dots = self.summaries.distances(query);

        for (block_id, dot) in dots.into_iter().enumerate() {
            if heap.len() == k && dot < heap_factor * heap.peek().distance.distance() {
                continue;
            }

            let packed_posting_block = &self.packed_postings
                [self.block_offsets[block_id]..self.block_offsets[block_id + 1]];

            self.evaluate_posting_block(
                evaluator,
                packed_posting_block,
                heap,
                visited,
                forward_index,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sort_and_search<S, E, QE, QC, QV>(
        &self,
        evaluator: &QE,
        query: &SparseVector1D<C, QueryValueFor<E>, QC, QV>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRangeDotProduct>,
        visited: &mut HashSet<usize>,
        forward_index: &S,
    ) where
        S: Dataset<E>,
        E: VectorEncoder<Distance = DotProduct>,
        QE: QueryEvaluator<E>,
        QC: AsRef<[C]>,
        QV: AsRef<[QueryValueFor<E>]>,
    {
        let dots = self.summaries.distances(query);
        let dots: Vec<_> = dots
            .into_iter()
            .enumerate()
            .sorted_unstable_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap())
            .collect();

        for (block_id, dot) in dots {
            if heap.len() == k && dot < heap_factor * heap.peek().distance.distance() {
                continue;
            }

            let packed_posting_block = &self.packed_postings
                [self.block_offsets[block_id]..self.block_offsets[block_id + 1]];

            self.evaluate_posting_block(
                evaluator,
                packed_posting_block,
                heap,
                visited,
                forward_index,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_posting_block<S, E, QE>(
        &self,
        evaluator: &QE,
        packed_posting_block: &[PackedPostingBlock],
        heap: &mut KHeap<ScoredRangeDotProduct>,
        visited: &mut HashSet<usize>,
        forward_index: &S,
    ) where
        S: Dataset<E>,
        E: VectorEncoder<Distance = DotProduct>,
        QE: QueryEvaluator<E>,
    {
        let mut iter = packed_posting_block.iter();
        let mut cur_pack = iter.next();

        while let Some(pack) = cur_pack {
            let next_pack = iter.next();
            if let Some(p) = next_pack {
                let range = p.unpack();
                forward_index.prefetch(range.start..(range.start + range.len()));
            }

            let range = pack.unpack();

            if visited.insert(range.start) {
                let vector = forward_index.get_by_range(range.clone());
                let distance = evaluator.compute_distance(vector);
                heap.push(ScoredRangeDotProduct { distance, range });
            }

            cur_pack = next_pack;
        }
    }

    pub(crate) fn build<S, E>(
        dataset: &S,
        postings: &[(ValueFor<E>, usize)],
        config: &Configuration,
    ) -> Self
    where
        S: Dataset<E>,
        E: SparseVectorEncoder,
        E: VectorEncoder<
                OutputComponentType = C,
                QueryComponentType = ComponentFor<E>,
                QueryValueType = f32,
            >,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
        C: std::hash::Hash,
        ValueFor<E>: PartialOrd,
    {
        let mut posting_list: Vec<_> = postings.iter().map(|(_, docid)| *docid).collect();

        let block_offsets = match config.blocking {
            BlockingStrategy::FixedSize { block_size } => {
                Self::fixed_size_blocking(&posting_list, block_size)
            }

            BlockingStrategy::RandomKmeans {
                centroid_fraction,
                min_cluster_size,
                clustering_algorithm,
            } => Self::blocking_with_random_kmeans(
                &mut posting_list,
                centroid_fraction,
                min_cluster_size,
                dataset,
                clustering_algorithm,
            ),
        };

        let quantizer = vectorium::PlainSparseQuantizer::<C, f32, vectorium::DotProduct>::new(
            dataset.input_dim(),
            dataset.input_dim(),
        );
        let mut summary = SparseDatasetGrowable::<
            vectorium::PlainSparseQuantizer<C, f32, vectorium::DotProduct>,
        >::new(quantizer);
        for (components, values) in
            block_offsets
                .array_windows()
                .map(|&[block_start, block_end]| {
                    let summary_vec = match config.summarization {
                        SummarizationStrategy::FixedSize { n_components } => {
                            Self::fixed_size_summary(
                                dataset,
                                &posting_list[block_start..block_end],
                                n_components,
                            )
                        }

                        SummarizationStrategy::EnergyPreserving {
                            summary_energy: fraction,
                        } => Self::energy_preserving_summary(
                            dataset,
                            &posting_list[block_start..block_end],
                            fraction,
                        ),
                    };
                    let (components, values): (Vec<C>, Vec<f32>) = summary_vec.into_iter().unzip();
                    (components, values)
                })
        {
            summary.push(SparseVector1D::new(components, values));
        }

        let packed_postings: Vec<_> = posting_list
            .iter()
            .map(|&doc_id| PackedPostingBlock::pack(dataset.range_from_id(doc_id as u64)))
            .collect();

        Self {
            packed_postings: packed_postings.into_boxed_slice(),
            block_offsets: block_offsets.into_boxed_slice(),
            summaries: QuantizedSummary::from(SparseDataset::from(summary)),
        }
    }

    fn fixed_size_blocking(posting_list: &[usize], block_size: usize) -> Vec<usize> {
        let mut result: Vec<_> = (0..posting_list.len() / block_size)
            .map(|i| i * block_size)
            .collect();
        if result.last() != Some(&posting_list.len()) {
            result.push(posting_list.len());
        }
        result
    }

    fn blocking_with_random_kmeans<S, E>(
        posting_list: &mut [usize],
        centroid_fraction: f32,
        min_cluster_size: usize,
        dataset: &S,
        clustering_algorithm: ClusteringAlgorithm,
    ) -> Vec<usize>
    where
        S: Dataset<E>,
        E: SparseVectorEncoder,
        E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
        E: VectorEncoder<QueryValueType = f32>,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
        ValueFor<E>: PartialOrd,
    {
        if posting_list.is_empty() {
            return Vec::new();
        }

        let n_centroids = ((centroid_fraction * posting_list.len() as f32) as usize).max(1);

        assert!(
            n_centroids <= u16::MAX as usize,
            "In the current implementation the number of centroids cannot be greater than u16::MAX. This is due that the quantized summary uses u16 to store the centroids ids (aka summaries ids).\n Please, decrease centroid_fraction!"
        );

        let mut reordered_posting_list = Vec::<_>::with_capacity(posting_list.len());
        let mut block_offsets = Vec::<_>::with_capacity(n_centroids + 1);

        // Build k-means clusters on the posting list
        let mut clusters = match clustering_algorithm {
            ClusteringAlgorithm::RandomKmeans {} => {
                do_random_kmeans_on_docids(posting_list, n_centroids, dataset, min_cluster_size)
            }
            ClusteringAlgorithm::RandomKmeansInvertedIndex {
                pruning_factor,
                doc_cut,
            } => do_random_kmeans_on_docids_ii_dot_product(
                posting_list,
                n_centroids,
                dataset,
                min_cluster_size,
                pruning_factor,
                doc_cut,
            ),
            ClusteringAlgorithm::RandomKmeansInvertedIndexApprox { doc_cut } => {
                do_random_kmeans_on_docids_ii_approx_dot_product(
                    posting_list,
                    n_centroids,
                    dataset,
                    min_cluster_size,
                    doc_cut,
                )
            }
        };

        clusters.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        for group in
            clusters.chunk_by(|&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| {
                centroid_id_a == centroid_id_b
            })
        {
            let _centroid_id = group[0].0;
            for &(_centroid_id, doc_id) in group {
                reordered_posting_list.push(doc_id);
            }
            block_offsets.push(reordered_posting_list.len());
        }

        posting_list.copy_from_slice(&reordered_posting_list);

        block_offsets
    }

    fn fixed_size_summary<S, E>(
        dataset: &S,
        block: &[usize],
        n_components: usize,
    ) -> Vec<(ComponentFor<E>, f32)>
    where
        S: Dataset<E>,
        E: VectorEncoder,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
        ComponentFor<E>: std::hash::Hash,
    {
        let mut hash = std::collections::HashMap::new();
        for &doc_id in block.iter() {
            let posting = dataset.get(doc_id as u64);
            let components = posting.components_as_slice();
            let values = posting.values_as_slice();
            for (&c, &v) in components.iter().zip(values) {
                let v = v.to_f32().unwrap();
                hash.entry(c)
                    .and_modify(|h| *h = if *h < v { v } else { *h })
                    .or_insert(v);
            }
        }

        hash.into_iter()
            .k_largest_by(n_components, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            })
            .sorted_unstable_by_key(|&(id, _)| id)
            .collect()
    }

    fn energy_preserving_summary<S, E>(
        dataset: &S,
        block: &[usize],
        fraction: f32,
    ) -> Vec<(ComponentFor<E>, f32)>
    where
        S: Dataset<E>,
        E: VectorEncoder,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
        ComponentFor<E>: std::hash::Hash,
    {
        let mut hash = std::collections::HashMap::new();
        for &doc_id in block.iter() {
            let posting = dataset.get(doc_id as u64);
            let components = posting.components_as_slice();
            let values = posting.values_as_slice();
            for (&c, &v) in components.iter().zip(values) {
                let v = v.to_f32().unwrap();
                hash.entry(c)
                    .and_modify(|h| *h = if *h < v { v } else { *h })
                    .or_insert(v);
            }
        }

        let mut components_values: Vec<_> = hash.into_iter().collect();

        components_values
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total_sum: f32 = components_values
            .iter()
            .map(|(_, x)| x.to_f32().unwrap())
            .sum();

        let until = total_sum * fraction;
        let mut acc = 0_f32;
        components_values
            .into_iter()
            .take_while_inclusive(|(_, v)| {
                acc += v.to_f32().unwrap();
                acc < until
            })
            .sorted_unstable_by_key(|&(id, _)| id)
            .collect()
    }
}
