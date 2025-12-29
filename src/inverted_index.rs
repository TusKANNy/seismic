use crate::utils::{KHeap, PackedPostingBlock, read_from_path, write_to_path};
use crate::QuantizedSummary;

use toolkit::BitFieldBoxed;
use toolkit::bitfield::BitFieldVec;

use vectorium::dataset::{ScoredRangeDotProduct, ScoredVectorDotProduct};
use vectorium::{
    ComponentType, Dataset, Distance, DotProduct, DotVByteFixedU8Quantizer, GrowableDataset,
    PackedDataset, PlainSparseDataset, PlainSparseQuantizer, QueryEvaluator, QueryVectorFor,
    SpaceUsage, SparseDataset, SparseDatasetGrowable, SparseVectorEncoder, SparseVector1D, ValueType,
    Vector1D, VectorEncoder,
};

use indicatif::ParallelProgressIterator;

use itertools::Itertools;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp;

use mem_dbg::{MemSize, SizeFlags};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use std::io::Result as IoResult;

type ComponentFor<E> = <E as VectorEncoder>::OutputComponentType;
type ValueFor<E> = <E as VectorEncoder>::OutputValueType;
type QueryValueFor<E> = <E as VectorEncoder>::QueryValueType;
type SparseEncodedVector<'a, E> =
    SparseVector1D<ComponentFor<E>, ValueFor<E>, &'a [ComponentFor<E>], &'a [ValueFor<E>]>;
type ScoredVectorFor = ScoredVectorDotProduct;

#[derive(Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct InvertedIndex<S, E>
where
    S: Dataset<E>,
    E: VectorEncoder,
{
    forward_index: S,
    #[serde(bound(
        serialize = "ComponentFor<E>: Serialize",
        deserialize = "ComponentFor<E>: DeserializeOwned"
    ))]
    posting_lists: Box<[PostingList<ComponentFor<E>>]>,
    config: Configuration,
    knn: Option<Knn>,
}

impl<S, E> SpaceUsage for InvertedIndex<S, E>
where
    S: Dataset<E> + SpaceUsage,
    E: VectorEncoder,
{
    fn space_usage_bytes(&self) -> usize {
        let forward = SpaceUsage::space_usage_bytes(&self.forward_index);

        let postings: usize = self
            .posting_lists
            .iter()
            .map(|list| list.space_usage_bytes())
            .sum();
        let knn_size = match &self.knn {
            Some(knn) => knn.space_usage_bytes(),
            None => 0,
        };

        forward + postings + knn_size
    }
}

/// This struct should contain every configuraion parameter for building the index
/// that doesn't need to be "managed" at query time.
/// Examples are the pruning strategy and the clustering strategy.
/// These can be chosen with a if at building time but there is no need to
/// make any choice at query time.
///
/// However, there are parameters that influence choices at query time.
/// To avoid branches or dynamic dispatching, this kind of parametrization are
/// selected with generic types.
/// An example is the quantization strategy. Based on the chosen
/// quantization strategy, we need to chose the right function to call while
/// computing the distance between vectors.
///
/// HERE WE COULD JUST HAVE A GENERIC IN THE SEARCH FUNCTION.
/// Have a non specialized search with a match and a call to the correct function!
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pruning: PruningStrategy,
    blocking: BlockingStrategy,
    summarization: SummarizationStrategy,
    knn: KnnConfiguration,
}

impl Configuration {
    pub fn pruning_strategy(mut self, pruning: PruningStrategy) -> Self {
        self.pruning = pruning;

        self
    }

    pub fn blocking_strategy(mut self, blocking: BlockingStrategy) -> Self {
        self.blocking = blocking;

        self
    }

    pub fn summarization_strategy(mut self, summarization: SummarizationStrategy) -> Self {
        self.summarization = summarization;

        self
    }

    pub fn knn(mut self, knn: KnnConfiguration) -> Self {
        self.knn = knn;

        self
    }
}

impl<S, E> InvertedIndex<S, E>
where
    S: Dataset<E> + Sync,
    E: VectorEncoder,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        assert!(
            list_id < self.posting_lists.len(),
            "Invalid list_id: {}",
            list_id
        );
        self.posting_lists[list_id]
            .get_all_doc_ranges()
            .into_iter()
            .map(|range| self.forward_index.id_from_range(range) as usize)
            .collect()
    }

    /// Help function to print the space usage of the index.
    pub fn print_space_usage_byte(&self) -> usize
    where
        S: SpaceUsage,
    {
        println!("Space Usage:");
        let forward = SpaceUsage::space_usage_bytes(&self.forward_index);
        println!("\tForward Index: {:} Bytes", forward);

        // Breakdown dettagliato delle posting lists
        let mut total_packed_postings = 0;
        let mut total_block_offsets = 0;
        let mut total_summaries = 0;

        for posting_list in self.posting_lists.iter() {
            total_packed_postings += SpaceUsage::space_usage_bytes(&posting_list.packed_postings);
            total_block_offsets += SpaceUsage::space_usage_bytes(&posting_list.block_offsets);
            total_summaries += posting_list.summaries.space_usage_bytes();
        }

        let postings_total = total_packed_postings + total_block_offsets + total_summaries;

        println!("\tPosting Lists: {:} Bytes", postings_total);
        println!(
            "\t  ├─ packed_postings: {:} Bytes ({:.2}%)",
            total_packed_postings,
            total_packed_postings as f64 / postings_total as f64 * 100.0
        );
        println!(
            "\t  ├─ block_offsets: {:} Bytes ({:.2}%)",
            total_block_offsets,
            total_block_offsets as f64 / postings_total as f64 * 100.0
        );
        println!(
            "\t  └─ summaries: {:} Bytes ({:.2}%)",
            total_summaries,
            total_summaries as f64 / postings_total as f64 * 100.0
        );

        let knn_size = match &self.knn {
            Some(knn) => knn.space_usage_bytes(),
            None => 0,
        };

        println!("\tKnn: {:} Bytes", knn_size);
        println!("\tTotal: {:} Bytes", forward + postings_total + knn_size);

        forward + postings_total + knn_size
    }

    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn search<QC, QV>(
        &self,
        query: &SparseVector1D<ComponentFor<E>, QueryValueFor<E>, QC, QV>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<ScoredVectorFor>
    where
        E: VectorEncoder<QueryComponentType = ComponentFor<E>, Distance = DotProduct>,
        QC: AsRef<[ComponentFor<E>]>,
        QV: AsRef<[QueryValueFor<E>]>,
        SparseVector1D<ComponentFor<E>, QueryValueFor<E>, QC, QV>: QueryVectorFor<E>,
    {
        let query_components = query.components_as_slice();
        let query_values = query.values_as_slice();

        // Assert that query components are sorted in case of using a mergsort like strategy for the dot product
        assert!(
            query_components.is_sorted(),
            "Query components must be sorted in ascending order."
        );

        let evaluator = self.forward_index.query_evaluator(query);

        let mut heap = KHeap::new(k);
        let mut visited = HashSet::with_capacity(query_cut.min(query_components.len()) * 5000); // TODO: 5000 should be n_postings

        // Evaluate the posting list only for the top score query terms
        let mut iter = query_components
            .iter()
            .zip(query_values)
            .k_largest_by(query_cut, |a, b| a.1.partial_cmp(b.1).unwrap());
        if first_sorted && let Some((&component_id, _value)) = iter.next() {
            self.posting_lists[component_id.as_()].sort_and_search(
                &evaluator,
                query,
                k,
                heap_factor,
                &mut heap,
                &mut visited,
                &self.forward_index,
            );
        }
        for (&component_id, _value) in iter {
            self.posting_lists[component_id.as_()].search(
                &evaluator,
                query,
                k,
                heap_factor,
                &mut heap,
                &mut visited,
                &self.forward_index,
            );
        }
        if n_knn > 0
            && let Some(knn) = self.knn.as_ref()
        {
            knn.refine(
                &evaluator,
                &mut heap,
                &mut visited,
                &self.forward_index,
                n_knn,
            );
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|scored| ScoredVectorDotProduct {
                distance: scored.distance,
                vector: self.forward_index.id_from_range(scored.range),
            })
            .collect()
    }

    /// `n_postings`: minimum number of postings to select for each component
    pub fn build(dataset: S, config: Configuration) -> Self
    where
        E: SparseVectorEncoder,
        E: VectorEncoder<
                QueryComponentType = ComponentFor<E>,
                QueryValueType = f32,
                Distance = DotProduct,
            >,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
    {
        print!("Distributing and pruning postings: ");
        let time = Instant::now();

        // Distribute pairs (score, doc_id) to corresponding components.
        //
        // The pruning strategy chosses how to filter out low-scoring postings.
        let inverted_pairs = match config.pruning {
            PruningStrategy::FixedSize { n_postings } => Self::fixed_pruning(&dataset, n_postings),
            PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction,
            } => Self::global_threshold_pruning(&dataset, n_postings, max_fraction),
            // Partial pruning for CoiThreshold is not implemented yet. Need to estimate the number of postings in the final list
            PruningStrategy::CoiThreshold {
                alpha: _,
                n_postings: _,
            } => {
                todo!()
            }
        };

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        println!("Number of posting lists: {}", inverted_pairs.len());

        let avg_list_length = inverted_pairs.iter().map(|l| l.len()).sum::<usize>() as f32
            / inverted_pairs.len() as f32;
        println!("Avg posting list length: {:.2}", avg_list_length);

        print!("Building summaries: ");
        let time = Instant::now();
        // Build summaries and blocks for each posting list
        let posting_lists: Vec<_> = inverted_pairs
            .par_iter()
            .progress_count(inverted_pairs.len() as u64)
            .enumerate()
            .map(|(_component_id, posting_list)| {
                PostingList::build(&dataset, posting_list, &config)
            })
            .collect();

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        if config.knn.nknn == 0 && config.knn.knn_path.is_none() {
            return Self {
                forward_index: dataset,
                posting_lists: posting_lists.into_boxed_slice(),
                config: config.clone(),
                knn: None,
            };
        }

        let me = InvertedIndex::<S, E> {
            forward_index: dataset,
            posting_lists: posting_lists.into_boxed_slice(),
            config: config.clone(),
            knn: None,
        };

        let time = Instant::now();
        let knn_config = config.knn.clone();
        let knn = if let Some(knn_path) = knn_config.knn_path {
            Knn::new_from_serialized(&knn_path, Some(knn_config.nknn))
        } else {
            Knn::new(&me, knn_config.nknn)
        };

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());
        Self {
            forward_index: me.forward_index,
            posting_lists: me.posting_lists,
            config,
            knn: Some(knn),
        }
    }

    /// Convert the `InvertedIndex`'s dataset to another one.
    pub fn from_inverted_index<T, ET>(inverted_index: InvertedIndex<T, ET>) -> Self
    where
        S: Dataset<E> + From<SparseDataset<E>>,
        E: SparseVectorEncoder<InputValueType = f32, InputComponentType = ComponentFor<E>>,
        E: vectorium::SpaceUsage,
        for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
        T: Dataset<ET>,
        ET: VectorEncoder<OutputComponentType = ComponentFor<E>, OutputValueType = f32>,
        for<'a> ET: VectorEncoder<
            EncodedVector<'a> = SparseVector1D<
                ComponentFor<E>,
                f32,
                &'a [ComponentFor<E>],
                &'a [f32],
            >,
        >,
    {
        // For datasets that are only used as bases for conversion, it would be better to store the id
        let old_packs: Box<_> = (0..inverted_index.forward_index.len())
            .map(|i| PackedPostingBlock::pack(inverted_index.forward_index.range_from_id(i as u64)))
            .collect();
        let input_dim = inverted_index.forward_index.input_dim();
        let output_dim = inverted_index.forward_index.output_dim();
        let quantizer = E::new(input_dim, output_dim);
        let mut growable = SparseDatasetGrowable::<E>::new(quantizer);
        for vector in inverted_index.forward_index.iter() {
            let components = vector.components_as_slice();
            let values = vector.values_as_slice();
            growable.push(SparseVector1D::new(components, values));
        }
        let frozen: SparseDataset<E> = growable.into();
        let new_dataset: S = frozen.into();
        let packs_map: HashMap<_, _> = old_packs
            .into_iter()
            .zip(
                (0..new_dataset.len())
                    .map(|i| PackedPostingBlock::pack(new_dataset.range_from_id(i as u64))),
            )
            .collect();

        let mut posting_lists = inverted_index.posting_lists;
        for posting in posting_lists.iter_mut() {
            for pack in posting.packed_postings.iter_mut() {
                *pack = packs_map[pack];
            }
        }

        Self {
            forward_index: new_dataset,
            posting_lists,
            config: inverted_index.config,
            knn: inverted_index.knn,
        }
    }

    /// Convenience function to build InvertedIndex using a certain dataset as a base, then convering the dataset.
    pub fn from_base_dataset<T, ET>(dataset: T, config: Configuration) -> Self
    where
        S: Dataset<E> + From<SparseDataset<E>>,
        E: SparseVectorEncoder<InputValueType = f32, InputComponentType = ComponentFor<E>>,
        E: vectorium::SpaceUsage,
        for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
        T: Dataset<ET> + Sync,
        ET: SparseVectorEncoder,
        ET: VectorEncoder<
                QueryComponentType = ComponentFor<ET>,
                QueryValueType = f32,
                Distance = DotProduct,
            >,
        ET: VectorEncoder<OutputComponentType = ComponentFor<E>, OutputValueType = f32>,
        for<'a> ET: VectorEncoder<
            EncodedVector<'a> = SparseVector1D<
                ComponentFor<E>,
                f32,
                &'a [ComponentFor<E>],
                &'a [f32],
            >,
        >,
    {
        let inverted_index = InvertedIndex::<T, ET>::build(dataset, config);
        Self::from_inverted_index(inverted_index)
    }

    /// Convert a `PlainSparseDataset<u16, f32>`-backed index into a DotVByte packed dataset.
    pub fn from_inverted_index_dotvbyte(
        inverted_index: InvertedIndex<
            PlainSparseDataset<u16, f32, DotProduct>,
            PlainSparseQuantizer<u16, f32, DotProduct>,
        >,
    ) -> InvertedIndex<PackedDataset<DotVByteFixedU8Quantizer>, DotVByteFixedU8Quantizer> {
        let old_packs: Box<_> = (0..inverted_index.forward_index.len())
            .map(|i| PackedPostingBlock::pack(inverted_index.forward_index.range_from_id(i as u64)))
            .collect();

        let new_dataset: PackedDataset<DotVByteFixedU8Quantizer> =
            inverted_index.forward_index.into();
        let packs_map: HashMap<_, _> = old_packs
            .into_iter()
            .zip(
                (0..new_dataset.len())
                    .map(|i| PackedPostingBlock::pack(new_dataset.range_from_id(i as u64))),
            )
            .collect();

        let mut posting_lists = inverted_index.posting_lists;
        for posting in posting_lists.iter_mut() {
            for pack in posting.packed_postings.iter_mut() {
                *pack = packs_map[pack];
            }
        }

        InvertedIndex {
            forward_index: new_dataset,
            posting_lists,
            config: inverted_index.config,
            knn: inverted_index.knn,
        }
    }

    // Add a precomputed knn graph to the index, limiting the number of neighbours to limit if it is not None
    //pub fn add_knn(&mut self, knn: Knn, limit: Option<usize>) {
    pub fn add_knn(&mut self, knn: Knn) {
        self.knn = Some(knn);
    }

    // Implementation of the pruning strategy that selects the top-`n_postings` from each posting list
    fn fixed_pruning(dataset: &S, n_postings: usize) -> Vec<Vec<(ValueFor<E>, usize)>>
    where
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
    {
        let mut inverted_pairs: Vec<KHeap<ScoredVectorDotProduct>> =
            vec![KHeap::new(n_postings); dataset.input_dim()];

        for (i, posting) in dataset.iter().enumerate() {
            let components = posting.components_as_slice();
            let values = posting.values_as_slice();
            for (&component, &value) in components.iter().zip(values) {
                let score = DotProduct::from(value.to_f32().unwrap());
                inverted_pairs[component.as_()].push(ScoredVectorDotProduct {
                    distance: score,
                    vector: i as u64,
                });
            }
        }

        inverted_pairs
            .into_iter()
            .map(|k| {
                k.into_sorted_vec()
                    .into_iter()
                    .map(|ScoredVectorDotProduct { distance, vector }| {
                        (
                            <ValueFor<E> as vectorium::FromF32>::from_f32_saturating(
                                distance.distance(),
                            ),
                            vector as usize,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    // Implementation of the pruning strategy that selects the top-x posting from each posting list where x is alpha times the number of postings in the list or max_n_posting if too big.
    #[allow(unused)]
    fn coi_pruning<V: ValueType>(
        inverted_pairs: &mut Vec<Vec<(V, usize)>>,
        alpha: f32,
        max_n_postings: usize,
    ) {
        inverted_pairs.par_iter_mut().for_each(|posting_list| {
            if posting_list.is_empty() {
                return;
            }
            posting_list.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let cur_n_postings =
                max_n_postings.min((posting_list.len() as f32 * alpha) as usize + 1);
            posting_list.truncate(cur_n_postings);

            posting_list.shrink_to_fit();
        })
    }

    // Implementation of the pruning strategy that selects the `n_postings*dim` top postings globally, with a limit of `n_postings*max_fraction` per posting.
    fn global_threshold_pruning(
        dataset: &S,
        n_postings: usize,
        max_fraction: f32,
    ) -> Vec<Vec<(ValueFor<E>, usize)>>
    where
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
    {
        let mut new_inverted_pairs = vec![Vec::new(); dataset.input_dim()];

        let tot_postings = dataset.input_dim() * n_postings; // overall number of postings to select

        let largest_entries = dataset
            .iter()
            .enumerate()
            .flat_map(|(i, posting)| {
                let components = posting.components_as_slice();
                let values = posting.values_as_slice();
                components
                    .iter()
                    .copied()
                    .zip(values.iter().copied())
                    .map(|p| (i, p))
                    .collect::<Vec<_>>()
            })
            .k_largest_by(tot_postings, |(_, (_, score_a)), (_, (_, score_b))| {
                score_a.partial_cmp(score_b).unwrap()
            });

        for (doc, (component, value)) in largest_entries {
            if new_inverted_pairs[component.as_()].len()
                < (n_postings as f32 * max_fraction) as usize
            {
                new_inverted_pairs[component.as_()].push((value, doc))
            };
        }

        new_inverted_pairs
    }

    /// Returns the sparse dataset
    pub fn dataset(&self) -> &S {
        &self.forward_index
    }

    /// Returns knn graph if present
    pub fn knn(&self) -> Option<&Knn> {
        self.knn.as_ref()
    }

    /// Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
    pub fn dim(&self) -> usize {
        self.forward_index.input_dim()
    }

    /// Returns the number of non-zero components in the dataset.
    pub fn nnz(&self) -> usize {
        self.forward_index.nnz()
    }

    /// Returns the number of vectors in the dataset
    pub fn len(&self) -> usize {
        self.forward_index.len()
    }

    /// Checks if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of neighbors in the knn graph, 0 if knn graph is not present.
    pub fn knn_len(&self) -> usize {
        match &self.knn {
            Some(knn) => knn.dim,
            None => 0,
        }
    }
}

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct PostingList<C: ComponentType> {
    packed_postings: Box<[PackedPostingBlock]>,
    block_offsets: Box<[usize]>,
    summaries: QuantizedSummary<C>,
}

impl<C: ComponentType> SpaceUsage for PostingList<C> {
    fn space_usage_bytes(&self) -> usize {
        SpaceUsage::space_usage_bytes(&self.packed_postings)
            + SpaceUsage::space_usage_bytes(&self.block_offsets)
            + self.summaries.space_usage_bytes()
    }
}

impl<C: ComponentType> PostingList<C> {
    pub fn get_all_doc_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::with_capacity(self.packed_postings.len());
        for pack in self.packed_postings.iter() {
            ranges.push(pack.unpack());
        }

        ranges
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search<S, E, QE, QC, QV>(
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

    // Sort summaries by dot product w.r.t. to the query. Useful only in the first list.
    // TODO: The reason the function is basically duplicated is to avoid pointless allocations if we don't need to sort.
    // Maybe there is a prettier way to do this without duplicating.
    #[allow(clippy::too_many_arguments)]
    pub fn sort_and_search<S, E, QE, QC, QV>(
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
                heap.push(ScoredRangeDotProduct {
                    distance,
                    range,
                });
            }

            cur_pack = next_pack;
        }
    }

    /// Gets a posting list already pruned and represents it by using a blocking
    /// strategy to partition postings into block and a summarization strategy to
    /// represents the summary of each block.
    pub fn build<S, E>(
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

    // ** Blocking strategies **

    fn fixed_size_blocking(posting_list: &[usize], block_size: usize) -> Vec<usize> {
        // Of course this strategy does not need offsets, but we are using them
        // just to have just a "universal" query search implementation
        let mut result: Vec<_> = (0..posting_list.len() / block_size)
            .map(|i| i * block_size)
            .collect();
        if result.last() != Some(&posting_list.len()) {
            result.push(posting_list.len());
        }
        result
    }

    // Panics if the number of centroids is greater than u16::MAX.
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
        let mut block_offsets = Vec::with_capacity(n_centroids);

        let clustering_results = match clustering_algorithm {
            ClusteringAlgorithm::RandomKmeans {} => crate::utils::do_random_kmeans_on_docids(
                posting_list,
                n_centroids,
                dataset,
                min_cluster_size,
            ),
            ClusteringAlgorithm::RandomKmeansInvertedIndex {
                pruning_factor,
                doc_cut,
            } => crate::utils::do_random_kmeans_on_docids_ii_dot_product(
                posting_list,
                n_centroids,
                dataset,
                min_cluster_size,
                pruning_factor,
                doc_cut,
            ),

            ClusteringAlgorithm::RandomKmeansInvertedIndexApprox { doc_cut } => {
                crate::utils::do_random_kmeans_on_docids_ii_approx_dot_product(
                    posting_list,
                    n_centroids,
                    dataset,
                    min_cluster_size,
                    doc_cut,
                )
            }
        };

        block_offsets.push(0);

        for group in clustering_results.chunk_by(
            // group by centroid_id
            |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| {
                centroid_id_a == centroid_id_b
            },
        ) {
            reordered_posting_list.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            block_offsets.push(reordered_posting_list.len());
        }

        posting_list.copy_from_slice(&reordered_posting_list);

        block_offsets
    }

    // ** Summarization strategies **

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
    {
        let mut hash: HashMap<ComponentFor<E>, f32> = HashMap::new();
        for &doc_id in block.iter() {
            // for each component_id, store the largest value seen so far
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
            // Take up to limit by decreasing scores
            .k_largest_by(n_components, |a, b| a.1.partial_cmp(&b.1).unwrap())
            // Sort by id to make binary search possible
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
    {
        let mut hash: HashMap<ComponentFor<E>, f32> = HashMap::new();
        for &doc_id in block.iter() {
            // for each component_id, store the largest value seen so far
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

        components_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
/// Represents the possible choices for the strategy used to prune the posting
/// lists at building time.
/// There are the following possible strategies:
/// - `Fixed  { n_postings: usize }`: Every posting list is pruned by only taking its top-`n_postings`
/// - `GlobalThreshold { n_postings: usize, max_fraction: f32 }`: Globally select the top postings, pruning the one with a smaller score. The number of posting kept in total is `n_postings*dim` (so that the average per posting list is `n_postings`), with a limit per posting list of `max_fraction*n_postings`.
/// - `CoiThreshold { alpha: f32, n_postings: usize }`: we prune each vector to preserve a fraction alpha of its L1 mass. Then, we prune the posting lists to have no more than n_postings: each. If n_postings is 0, then we skip the last step.
pub enum PruningStrategy {
    FixedSize {
        n_postings: usize,
    },
    GlobalThreshold {
        n_postings: usize,
        max_fraction: f32, // limits the length of each posting list to max_fraction*n_postings
    },
    CoiThreshold {
        alpha: f32,
        n_postings: usize,
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        Self::GlobalThreshold {
            n_postings: 3500,
            max_fraction: 1.5,
        }
    }
}

// we need this because clap does not support enums with associated values
#[derive(clap::ValueEnum, Default, Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PruningStrategyClap {
    FixedSize,
    #[default]
    GlobalThreshold,
    CoiThreshold,
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum BlockingStrategy {
    FixedSize {
        block_size: usize,
    },

    RandomKmeans {
        centroid_fraction: f32,
        min_cluster_size: usize,
        clustering_algorithm: ClusteringAlgorithm,
    },
}

impl Default for BlockingStrategy {
    fn default() -> Self {
        BlockingStrategy::RandomKmeans {
            centroid_fraction: 0.1,
            min_cluster_size: 2,
            clustering_algorithm: ClusteringAlgorithm::default(),
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SummarizationStrategy {
    FixedSize { n_components: usize },
    EnergyPreserving { summary_energy: f32 },
}

impl Default for SummarizationStrategy {
    fn default() -> Self {
        Self::EnergyPreserving {
            summary_energy: 0.4,
        }
    }
}

// we need this because clap does not support enums with associated values
#[derive(clap::ValueEnum, Default, Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClusteringAlgorithmClap {
    RandomKmeans,
    RandomKmeansInvertedIndex,
    #[default]
    RandomKmeansInvertedIndexApprox,
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    RandomKmeans {}, // This is the original implementation of RandomKmeans, the algorithm is very slow for big datasets
    RandomKmeansInvertedIndex { pruning_factor: f32, doc_cut: usize }, // call crate::utils::do_random_kmeans_on_docids_ii_dot_product
    RandomKmeansInvertedIndexApprox { doc_cut: usize }, // call crate::utils::do_random_kmeans_on_docids_ii_approx_dot_product
}

impl Default for ClusteringAlgorithm {
    fn default() -> Self {
        Self::RandomKmeansInvertedIndexApprox { doc_cut: 15 }
    }
}

#[derive(PartialEq, Default, Debug, Clone, Serialize, Deserialize)]
pub struct KnnConfiguration {
    nknn: usize,
    knn_path: Option<String>,
}

impl KnnConfiguration {
    pub fn new(nknn: usize, knn_path: Option<String>) -> Self {
        KnnConfiguration { nknn, knn_path }
    }
}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize, Default)]
pub struct Knn {
    n_vecs: usize,
    dim: usize,
    neighbours: BitFieldBoxed,
}

impl SpaceUsage for Knn {
    fn space_usage_bytes(&self) -> usize {
        self.neighbours.mem_size(SizeFlags::empty())
            + SpaceUsage::space_usage_bytes(&self.n_vecs)
            + SpaceUsage::space_usage_bytes(&self.dim)
    }
}

impl Knn {
    pub fn new<S, E>(index: &InvertedIndex<S, E>, dim: usize) -> Self
    where
        S: Dataset<E> + Sync,
        E: SparseVectorEncoder,
        E: VectorEncoder<QueryComponentType = ComponentFor<E>, Distance = DotProduct>,
        E: VectorEncoder<QueryValueType = f32>,
        for<'a> <E as VectorEncoder>::EncodedVector<'a>:
            Vector1D<Component = ComponentFor<E>, Value = ValueFor<E>>,
    {
        const KNN_QUERY_CUT: usize = 10;
        const KNN_HEAP_FACTOR: f32 = 0.7;

        let n_vecs = index.len();
        print!("Computing kNN: ");
        let docs_search_results: Vec<_> = (0..index.forward_index.len())
            .into_par_iter()
            .progress_count(index.forward_index.len() as u64)
            .map(|my_doc_id| {
                let vec = index.forward_index.get(my_doc_id as u64);
                let components = vec.components_as_slice();
                let values = vec.values_as_slice();
                let f32_values: Vec<_> = values.iter().map(|v| v.to_f32().unwrap()).collect();
                let components = components.to_vec();

                let query = SparseVector1D::new(components, f32_values);
                index
                    .search(
                        &query,
                        dim + 1, // +1 to filter the document itself if present
                        KNN_QUERY_CUT,
                        KNN_HEAP_FACTOR,
                        0,
                        false,
                    )
                    .into_iter()
                    .filter(|result| result.vector as usize != my_doc_id) // remove the document itself
                    .take(dim)
                    .collect::<Vec<_>>()
            })
            .collect();

        let neighbours: Vec<u64> = docs_search_results
            .into_iter()
            .flat_map(|r| r.into_iter())
            .map(|result| result.vector)
            .collect();

        let bitfield = BitFieldBoxed::from(neighbours);

        Self {
            n_vecs,
            dim,
            neighbours: bitfield,
        }
    }

    pub fn new_from_serialized(path: &str, limit: Option<usize>) -> Self {
        println!("Reading KNN from file: {:}", path);
        let knn: Knn = read_from_path(path).unwrap();

        println!("Number of vectors: {:}", knn.n_vecs);
        println!("Number of neighbors in the file: {:}", knn.dim);

        let nknn = limit.unwrap_or(knn.dim);

        assert!(
            nknn <= knn.dim,
            "The number of neighbors to include for each vector of the dataset can't be greater than the number of neighbours in the precomputed knn file."
        );

        if nknn == knn.dim {
            return knn;
        } else {
            println!("We only take {:} neighbors per element!", nknn);
        }

        let mut neighbours = BitFieldVec::with_capacity(knn.n_vecs, knn.neighbours.field_width());

        for id in 0..knn.n_vecs {
            let base_pos = id * knn.dim;
            for i in 0..nknn {
                let neighbor = knn.neighbours.get(base_pos + i).unwrap();
                neighbours.push(neighbor);
            }
        }

        Knn {
            n_vecs: knn.n_vecs,
            dim: nknn,
            neighbours: neighbours.convert_into(),
        }
    }

    pub fn serialize(&self, output_file: &str) -> IoResult<()> {
        let path = output_file.to_string() + ".knn.seismic";
        println!("Saving ... {}", path.as_str());
        write_to_path(self, path.as_str()).unwrap();
        Ok(())
    }

    #[inline]
    pub(crate) fn refine<S, E, QE>(
        &self,
        evaluator: &QE,
        heap: &mut KHeap<ScoredRangeDotProduct>,
        visited: &mut HashSet<usize>,
        forward_index: &S,
        in_n_knn: usize,
    ) where
        S: Dataset<E>,
        E: VectorEncoder<Distance = DotProduct>,
        QE: QueryEvaluator<E>,
    {
        let n_knn = cmp::min(self.dim, in_n_knn);

        let neighbours: Vec<_> = heap.clone().into_sorted_vec();

        for ScoredRangeDotProduct {
            distance: _distance,
            range,
        } in neighbours.into_iter()
        {
            let id = forward_index.id_from_range(range.clone()) as usize;
            let base_pos = id * self.dim;

            for i in 0..n_knn {
                // SAFETY: we are sure the position is valid
                let neighbour = unsafe { self.neighbours.get_unchecked(base_pos + i) } as usize;

                let range = forward_index.range_from_id(neighbour as u64);

                if visited.insert(range.start) {
                    let vector = forward_index.get_by_range(range.clone());
                    let distance = evaluator.compute_distance(vector);
                    heap.push(ScoredRangeDotProduct { distance, range });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::{
        DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        SparseVector1D,
    };

    // Test pushing empty vectors.
    #[test]
    fn test_empty_vectors() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
        let mut dataset = PlainSparseDatasetGrowable::new(quantizer);

        // Push a single vector
        dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        // Push another vector
        let c = Vec::new();
        let v = Vec::new();

        dataset.push(SparseVector1D::new(c.clone(), v.clone()));
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        dataset.push(SparseVector1D::new(c, v));
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        // Push a fourth vector
        dataset.push(SparseVector1D::new(
            vec![0, 1, 2, 3],
            vec![1.0, 2.0, 3.0, 4.0],
        ));
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 7);

        let dataset: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();

        let index = InvertedIndex::build(dataset, Configuration::default());
        assert_eq!(index.len(), 4);
        assert_eq!(index.dim(), 5);
        assert_eq!(index.nnz(), 7);

        let query = SparseVector1D::new(vec![0_u16, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]);
        let results = index.search(&query, 10, 5, 0.7, 0, false);

        assert_eq!(results.len(), 2); // Empty vectors are never retrieved becasue they do not belong to any posting list!
        assert_eq!(results[0].vector, 3);
        assert_eq!(results[1].vector, 0);
    }
}
