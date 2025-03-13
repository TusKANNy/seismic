use crate::distances::{dot_product_dense_sparse, dot_product_with_merge};
use crate::sparse_dataset::{SparseDatasetIter, SparseDatasetMut};
use crate::topk_selectors::{HeapFaiss, OnlineTopKSelector};
use crate::utils::prefetch_read_NTA;
use crate::{DataType, QuantizedSummary, SpaceUsage, SparseDataset};

use indicatif::ParallelProgressIterator;

use qwt::SpaceUsage as SpaceUsageQwt;
use qwt::{BitVector, BitVectorMut};
use std::fs;

use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp;

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use std::io::Result as IoResult;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndex<T>
where
    T: DataType,
{
    forward_index: SparseDataset<T>,
    posting_lists: Box<[PostingList]>,
    config: Configuration,
    knn: Option<Knn>,
}

impl<T> SpaceUsage for InvertedIndex<T>
where
    T: DataType,
{
    fn space_usage_byte(&self) -> usize {
        let forward = self.forward_index.space_usage_byte();

        let postings: usize = self
            .posting_lists
            .iter()
            .map(|list| list.space_usage_byte())
            .sum();
        let knn_size = match &self.knn {
            Some(knn) => knn.space_usage_byte(),
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
/// Howerver, there are parameters that influence choices at query time.
/// To avoid branches or dynamic dispatching, this kind of parametrizaton are
/// selected with generic types.
/// An example is the quantization strategy. Based on the chosen
/// quantization strategy, we need to chose the right function to call while
/// computing the distance between vectors.
///
/// HERE WE COULD JUST HAVE A GENERIC IN THE SEARCH FUNCTION.
/// Have a non specilized search with a match and a call to the correct function!

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pruning: PruningStrategy,
    blocking: BlockingStrategy,
    summarization: SummarizationStrategy,
    knn: KnnConfiguration,
    batched_indexing: Option<usize>,
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

    pub fn batched_indexing(mut self, batched_indexing: Option<usize>) -> Self {
        self.batched_indexing = batched_indexing;

        self
    }
}

const THRESHOLD_BINARY_SEARCH: usize = 10;

impl<T> InvertedIndex<T>
where
    T: PartialOrd + DataType,
{
    /// Help function to print the space usage of the index.
    pub fn print_space_usage_byte(&self) -> usize {
        println!("Space Usage:");
        let forward = self.forward_index.space_usage_byte();
        println!("\tForward Index: {:} Bytes", forward);
        let postings: usize = self
            .posting_lists
            .iter()
            .map(|list| list.space_usage_byte())
            .sum();

        let knn_size = match &self.knn {
            Some(knn) => knn.space_usage_byte(),
            None => 0,
        };

        println!("\tPosting Lists: {:} Bytes", postings);
        println!("\tKnn: {:} Bytes", knn_size);
        println!("\tTotal: {:} Bytes", forward + postings + knn_size);

        forward + postings + knn_size
    }

    #[allow(clippy::too_many_arguments)]
    #[must_use]
    #[inline]
    pub fn search(
        &self,
        query_components: &[u16],
        query_values: &[f32],
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<(f32, usize)> {
        let mut query = vec![0.0; self.dim()];

        for (&i, &v) in query_components.iter().zip(query_values) {
            query[i as usize] = v;
        }
        let mut heap = HeapFaiss::new(k);
        let mut visited = HashSet::with_capacity(query_cut * 5000); // 5000 should be n_postings

        // Sort query terms by score and evaluate the posting list only for the top ones
        for (i, (&component_id, &_value)) in query_components
            .iter()
            .zip(query_values)
            .sorted_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap())
            .take(query_cut)
            .enumerate()
        {
            self.posting_lists[component_id as usize].search(
                &query,
                query_components,
                query_values,
                k,
                heap_factor,
                &mut heap,
                &mut visited,
                &self.forward_index,
                i == 0 && first_sorted,
            );
        }
        if let Some(knn) = self.knn.as_ref() {
            if n_knn > 0 {
                knn.refine(&query, &mut heap, &mut visited, &self.forward_index, n_knn);
            }
        }

        heap.topk()
            .iter()
            .map(|&(dot, offset)| (dot.abs(), self.forward_index.offset_to_id(offset)))
            .collect()
    }

    /// `n_postings`: minimum number of postings to select for each component
    pub fn build(dataset: SparseDataset<T>, config: Configuration) -> Self {
        // Distribute pairs (score, doc_id) to corresponding components for each chunk.
        // We use pairs because later each posting list will be sorted by score
        // by the pruning strategy.
        // The pruning strategy is applied to partial results, for Global Threshold strategy
        // the final fixed pruning is done only when all chunks have been parsed

        print!("Distributing and pruning postings: ");
        let time = Instant::now();

        let mut inverted_pairs = Vec::with_capacity(dataset.dim());
        let mut chunk_inv_pairs = Vec::with_capacity(dataset.dim());

        for _ in 0..dataset.dim() {
            inverted_pairs.push(Vec::new());
            chunk_inv_pairs.push(Vec::new());
        }

        let chunk_size = config.batched_indexing.unwrap_or(dataset.len());

        for chunk in &dataset.iter().enumerate().chunks(chunk_size) {
            for (_, (doc_id, (components, values))) in chunk.enumerate() {
                for (&c, &score) in components.iter().zip(values) {
                    chunk_inv_pairs[c as usize].push((score, doc_id));
                }
            }

            // If not batched indexing, chunk_inv_pairs already contain all the pairs
            if chunk_size == dataset.len() {
                inverted_pairs = chunk_inv_pairs;
            } else {
                // Copy the pairs of the current chunk in the partial results
                for (c, chunk_pairs) in chunk_inv_pairs.iter().enumerate() {
                    for (score, doc_id) in chunk_pairs.iter() {
                        inverted_pairs[c].push((*score, *doc_id));
                    }
                }
            }

            // Pruning on partial result
            match config.pruning {
                PruningStrategy::FixedSize { n_postings } => {
                    Self::fixed_pruning(&mut inverted_pairs, n_postings);
                }
                PruningStrategy::GlobalThreshold { n_postings, .. } => {
                    Self::global_threshold_pruning(&mut inverted_pairs, n_postings);
                }
            }

            chunk_inv_pairs = Vec::with_capacity(dataset.dim());
            for _ in 0..dataset.dim() {
                chunk_inv_pairs.push(Vec::new());
            }
        }

        // Final pruning
        match config.pruning {
            PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction,
            } => {
                Self::fixed_pruning(
                    &mut inverted_pairs,
                    (n_postings as f32 * max_fraction) as usize,
                );
            }
            _ => {}
        }

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        println!("Number of posting lists: {}", inverted_pairs.len());

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

        let me = Self {
            forward_index: dataset,
            posting_lists: posting_lists.into_boxed_slice(),
            config: config.clone(),
            knn: None,
        };

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        if config.knn.nknn == 0 && config.knn.knn_path.is_none() {
            //if config.knn.nknn == 0 {
            return me;
        }

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

    // Add a precomputed knn graph to the index, limiting the number of neighbours to limit if it is not None
    //pub fn add_knn(&mut self, knn: Knn, limit: Option<usize>) {
    pub fn add_knn(&mut self, knn: Knn) {
        self.knn = Some(knn);
    }

    // Implementation of the pruning strategy that selects the top-`n_postings` from each posting list
    fn fixed_pruning(inverted_pairs: &mut Vec<Vec<(T, usize)>>, n_postings: usize) {
        inverted_pairs.par_iter_mut().for_each(|posting_list| {
            posting_list.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            posting_list.truncate(n_postings);

            posting_list.shrink_to_fit();
        })
    }

    // Implementation of the NEW pruning strategy that selects a threshold such that survives on average `n_postings` for each posting list
    // In the new version, documents with scores equal to the threshold are included, without exceeding the threshold of 10% of expected postings
    fn global_threshold_pruning(inverted_pairs: &mut [Vec<(T, usize)>], n_postings: usize) {
        let tot_postings = inverted_pairs.len() * n_postings; // overall number of postings to select

        const EQUALITY_THRESHOLD: usize = 10;
        let max_eq_postings = EQUALITY_THRESHOLD * tot_postings / 100; //maximium number of posting with score equal to the threshold

        // for every posting we create the tuple <score, docid, id_posting_list>
        let mut postings = Vec::<(T, usize, u16)>::new();
        for (id, posting_list) in inverted_pairs.iter_mut().enumerate() {
            for (score, docid) in posting_list.iter() {
                postings.push((*score, *docid, id as u16));
            }
            posting_list.clear();
        }

        let tot_postings = tot_postings.min(postings.len() - 1);

        // To ensure that executions with different batch sizes provide the same result, the comparison criterion considers both the score and the doc_id
        let (_, (t_score, _, _), leq) =
            postings.select_nth_unstable_by(tot_postings, |a, b| b.partial_cmp(&a).unwrap());
        // All postings with scores equal to the threshold are added, up to max_eq_postings
        let (eq_pairs, _): (Vec<(T, usize, u16)>, _) = leq.iter().partition(|p| p.0 == *t_score);
        for (score, docid, id_postings) in eq_pairs.iter().take(max_eq_postings) {
            inverted_pairs[*id_postings as usize].push((*score, *docid));
        }

        if eq_pairs.len() > max_eq_postings {
            println!("A lot of entries have the same value. {} have been pruned, for more info look at DOC_REFERENCE", eq_pairs.len()-max_eq_postings);
        }

        for (score, docid, id_posting) in postings.into_iter().take(tot_postings) {
            inverted_pairs[id_posting as usize].push((score, docid));
        }
    }

    /// Returns the sparse dataset
    pub fn dataset(&self) -> &SparseDataset<T> {
        &self.forward_index
    }

    /// Returns knn graph if present
    pub fn knn(&self) -> Option<&Knn> {
        self.knn.as_ref()
    }

    /// Returns an iterator over the underlying SparseDataset
    #[must_use]
    pub fn iter(&self) -> SparseDatasetIter<T> {
        self.forward_index.iter()
    }

    /// Returns (offset, len) of the "id"-th document
    #[must_use]
    #[inline]
    pub fn id_to_offset_len(&self, id: usize) -> (usize, usize) {
        self.forward_index.id_to_offset_len(id)
    }

    /// Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.forward_index.dim()
    }

    /// Returns the number of non-zero components in the dataset.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.forward_index.nnz()
    }

    /// Returns the number of vectors in the dataset
    #[must_use]
    pub fn len(&self) -> usize {
        self.forward_index.len()
    }

    /// Checks if the dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forward_index.len() == 0
    }

    /// Returns the number of neighbors in the knn graph, 0 if knn graph is not present.
    #[must_use]
    pub fn knn_len(&self) -> usize {
        match &self.knn {
            Some(knn) => knn.d,
            None => 0,
        }
    }
}

// Instead of string doc_ids we store their offsets in the forward_index and the lengths of the vectors
// This allows us to save the random acceses that would be needed to access exactly these values from the
// forward index. The values of each doc are packed into a single u64 in `packed_postings`. We use 48 bits for the offset and 16 bits for the lenght. This choice limits the size of the dataset to be 1<<48-1.
// We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct PostingList {
    // postings: Box<[usize]>,
    packed_postings: Box<[u64]>,
    block_offsets: Box<[usize]>,
    // summaries: SparseDataset<f16>,
    summaries: QuantizedSummary,
}

impl SpaceUsage for PostingList {
    fn space_usage_byte(&self) -> usize {
        SpaceUsage::space_usage_byte(&self.packed_postings) +
        //self.packed_postings.space_usage_byte()
        SpaceUsage::space_usage_byte(&self.block_offsets)
        //    + self.block_offsets.space_usage_byte()
            + self.summaries.space_usage_byte()
    }
}

impl PostingList {
    #[inline]
    pub fn pack_offset_len(offset: usize, len: usize) -> u64 {
        ((offset as u64) << 16) | (len as u64)
    }

    #[inline]
    pub fn unpack_offset_len(pack: u64) -> (usize, usize) {
        ((pack >> 16) as usize, (pack & (u16::MAX as u64)) as usize)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn search<T>(
        &self,
        query: &[f32],
        query_components: &[u16],
        query_values: &[f32],
        k: usize,
        heap_factor: f32,
        heap: &mut HeapFaiss,
        visited: &mut HashSet<usize>,
        forward_index: &SparseDataset<T>,
        sort_summaries: bool,
    ) where
        T: DataType,
    {
        let mut blocks_to_evaluate: Vec<&[u64]> = Vec::new();

        let dots = self
            .summaries
            .distances_iter(query_components, query_values);

        let mut indexed_dots: Vec<(usize, f32)> = dots.enumerate().collect();

        // Sort summaries by dot product w.r.t. to the query. Useful only in the first list.
        if sort_summaries {
            indexed_dots.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        }

        for &(block_id, dot) in indexed_dots.iter() {
            if heap.len() == k && dot < -heap_factor * heap.top() {
                continue;
            }

            let packed_posting_block = &self.packed_postings
                [self.block_offsets[block_id]..self.block_offsets[block_id + 1]];

            if blocks_to_evaluate.len() == 1 {
                for cur_packed_posting in blocks_to_evaluate.iter() {
                    self.evaluate_posting_block(
                        query,
                        query_components,
                        query_values,
                        cur_packed_posting,
                        heap,
                        visited,
                        forward_index,
                    );
                }
                blocks_to_evaluate.clear();
            }

            for i in (0..packed_posting_block.len()).step_by(8) {
                prefetch_read_NTA(packed_posting_block, i);
            }

            blocks_to_evaluate.push(packed_posting_block);
        }

        for cur_packed_posting in blocks_to_evaluate.iter() {
            self.evaluate_posting_block(
                query,
                query_components,
                query_values,
                cur_packed_posting,
                heap,
                visited,
                forward_index,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn evaluate_posting_block<T>(
        &self,
        query: &[f32],
        query_term_ids: &[u16],
        query_values: &[f32],
        packed_posting_block: &[u64],
        heap: &mut HeapFaiss,
        visited: &mut HashSet<usize>,
        forward_index: &SparseDataset<T>,
    ) where
        T: DataType,
    {
        let (mut prev_offset, mut prev_len) = Self::unpack_offset_len(packed_posting_block[0]);

        for &pack in packed_posting_block.iter().skip(1) {
            let (offset, len) = Self::unpack_offset_len(pack);
            forward_index.prefetch_vec_with_offset(offset, len);

            if !visited.contains(&prev_offset) {
                let (v_components, v_values) = forward_index.get_with_offset(prev_offset, prev_len);
                //let distance = dot_product_dense_sparse(query, v_components, v_values);
                let distance = if query_term_ids.len() < THRESHOLD_BINARY_SEARCH {
                    //dot_product_with_binary_search(
                    dot_product_with_merge(query_term_ids, query_values, v_components, v_values)
                } else {
                    dot_product_dense_sparse(query, v_components, v_values)
                };

                visited.insert(prev_offset);
                heap.push_with_id(-1.0 * distance, prev_offset);
            }

            prev_offset = offset;
            prev_len = len;
        }

        if visited.contains(&prev_offset) {
            return;
        }

        let (v_components, v_values) = forward_index.get_with_offset(prev_offset, prev_len);
        let distance = if query_term_ids.len() < THRESHOLD_BINARY_SEARCH {
            //dot_product_with_binary_search(
            dot_product_with_merge(query_term_ids, query_values, v_components, v_values)
        } else {
            dot_product_dense_sparse(query, v_components, v_values)
        };

        visited.insert(prev_offset);
        heap.push_with_id(-1.0 * distance, prev_offset);
    }

    /// Gets a posting list already pruned and represents it by using a blocking
    /// strategy to partition postings into block and a summarization strategy to
    /// represents the summary of each block.
    pub fn build<T>(
        dataset: &SparseDataset<T>,
        postings: &[(T, usize)],
        config: &Configuration,
    ) -> Self
    where
        T: PartialOrd + DataType,
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

        let mut summaries = SparseDatasetMut::<T>::new();

        for block_range in block_offsets.windows(2) {
            let (components, values) = match config.summarization {
                SummarizationStrategy::FixedSize { n_components } => Self::fixed_size_summary(
                    dataset,
                    &posting_list[block_range[0]..block_range[1]],
                    n_components,
                ),

                SummarizationStrategy::EnergyPreserving {
                    summary_energy: fraction,
                } => Self::energy_preserving_summary(
                    dataset,
                    &posting_list[block_range[0]..block_range[1]],
                    fraction,
                ),
            };

            summaries.push(&components, &values);
        }

        let packed_postings: Vec<_> = posting_list
            .iter()
            .map(|doc_id| {
                Self::pack_offset_len(dataset.vector_offset(*doc_id), dataset.vector_len(*doc_id))
            })
            .collect();

        Self {
            packed_postings: packed_postings.into_boxed_slice(),
            block_offsets: block_offsets.into_boxed_slice(),
            summaries: QuantizedSummary::from(SparseDataset::<T>::from(summaries)), // Avoid to do from SparseDatasetMut. Problably fixed when moving to kANNolo SparseDataset
        }
    }

    // ** Blocking strategies **

    fn fixed_size_blocking(posting_list: &[usize], block_size: usize) -> Vec<usize> {
        // of course this strategy would not need offsets, but we are using them
        // just to have just one, "universal" query search implementation
        let mut block_offsets: Vec<_> = (0..posting_list.len() / block_size)
            .map(|i| i * block_size)
            .collect();

        block_offsets.push(posting_list.len());
        block_offsets
    }

    // Panics if the number of centroids is greater than u16::MAX.
    fn blocking_with_random_kmeans<T: DataType>(
        posting_list: &mut [usize],
        centroid_fraction: f32,
        min_cluster_size: usize,
        dataset: &SparseDataset<T>,
        clustering_algorithm: ClusteringAlgorithm,
    ) -> Vec<usize> {
        if posting_list.is_empty() {
            return Vec::new();
        }

        let n_centroids = ((centroid_fraction * posting_list.len() as f32) as usize).max(1);

        assert!(n_centroids <= u16::MAX as usize,  "In the current implementation the number of centroids cannot be greater than u16::MAX. This is due that the quantizied summary uses u16 to store the centroids ids (aka summaries ids).\n Please, decrease centroid_fraction!");

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

        assert_eq!(reordered_posting_list.len(), posting_list.len());
        posting_list.copy_from_slice(&reordered_posting_list);

        block_offsets
    }

    // ** Summarization strategies **

    fn fixed_size_summary<T>(
        dataset: &SparseDataset<T>,
        block: &[usize],
        n_components: usize,
    ) -> (Vec<u16>, Vec<T>)
    where
        T: PartialOrd + DataType,
    {
        let mut hash = HashMap::new();
        for &doc_id in block.iter() {
            // for each component_id, store the largest value seen so far
            for (&c, &v) in dataset.iter_vector(doc_id) {
                hash.entry(c)
                    .and_modify(|h| *h = if *h < v { v } else { *h })
                    .or_insert(v);
            }
        }

        let mut components_values: Vec<_> = hash.iter().collect();

        // First sort by decreasing scores, then take only up to LIMIT and sort by component_id
        components_values.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        components_values.truncate(n_components);

        components_values.sort_unstable_by(|a, b| a.0.cmp(b.0)); // sort by id to make binary search possible

        let components: Vec<_> = components_values
            .iter()
            .map(|(&component_id, _score)| component_id)
            .collect();

        let values: Vec<_> = components.iter().copied().map(|k| hash[&k]).collect();

        (components, values)
    }

    fn energy_preserving_summary<T>(
        dataset: &SparseDataset<T>,
        block: &[usize],
        fraction: f32,
    ) -> (Vec<u16>, Vec<T>)
    where
        T: PartialOrd + DataType,
    {
        let mut hash = HashMap::new();
        for &doc_id in block.iter() {
            // for each component_id, store the largest value seen so far
            for (&c, &v) in dataset.iter_vector(doc_id) {
                hash.entry(c)
                    .and_modify(|h| *h = if *h < v { v } else { *h })
                    .or_insert(v);
            }
        }

        let mut components_values: Vec<_> = hash.iter().collect();

        components_values.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let total_sum = components_values
            .iter()
            .fold(0_f32, |sum, (_, &x)| sum + x.to_f32().unwrap());

        let mut term_ids = Vec::new();
        let mut values = Vec::new();
        let mut acc = 0_f32;
        for (&tid, &v) in components_values.iter() {
            acc += v.to_f32().unwrap();
            term_ids.push(tid);
            values.push(v);
            if (acc / total_sum) > fraction {
                break;
            }
        }
        term_ids.sort();
        let values: Vec<T> = term_ids.iter().copied().map(|k| hash[&k]).collect();
        (term_ids, values)
    }
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
/// Represents the possible choices for the strategy used to prune the posting
/// lists at building time.
/// There are the following possible strategies:
/// - `Fixed  { n_postings: usize }`: Every posting list is pruned by taking its top-`n_postings`
/// - `GlobalThreshold { n_postings: usize, max_fraction: f32 }`: We globally select a threshold and we prune all the postings with smaller score. The threshold is chosen so that every posting list has `n_postings` on average. We limit the number of postings per list to `max_fraction*n_postings`.
pub enum PruningStrategy {
    FixedSize {
        n_postings: usize,
    },
    GlobalThreshold {
        n_postings: usize,
        max_fraction: f32, // limits the length of each posting list to max_fraction*n_postings
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        Self::FixedSize { n_postings: 3500 }
    }
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
    RandomKmeans {}, // This is the original implementation of RandomKmeans, the algorith is very slow for big datasets
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
    d: usize,
    neighbours: BitVector,
    nbits: usize,
}

impl SpaceUsage for Knn {
    fn space_usage_byte(&self) -> usize {
        self.neighbours.space_usage_byte()
            + SpaceUsage::space_usage_byte(&self.n_vecs)
            + SpaceUsage::space_usage_byte(&self.d)
            + SpaceUsage::space_usage_byte(&self.nbits)
    }
}

impl Knn {
    #[must_use]
    pub fn new<T: PartialOrd + DataType>(index: &InvertedIndex<T>, d: usize) -> Self {
        const KNN_QUERY_CUT: usize = 10;
        const KNN_HEAP_FACTOR: f32 = 0.7;

        let n_vecs = index.len();
        print!("Computing kNN: ");
        let docs_search_results: Vec<_> = index
            .forward_index
            .par_iter()
            .progress_count(index.forward_index.len() as u64)
            .enumerate()
            .map(|(my_doc_id, (components, values))| {
                let f32_values: Vec<f32> = values.iter().map(|v| v.to_f32().unwrap()).collect();

                index
                    .search(
                        components,
                        &f32_values,
                        d + 1, // +1 to filter the document itself if present
                        KNN_QUERY_CUT,
                        KNN_HEAP_FACTOR,
                        0,
                        false,
                    )
                    .iter()
                    .map(|(distance, doc_id)| (*distance, *doc_id))
                    .filter(|(_distance, doc_id)| *doc_id != my_doc_id) // remove the document itself
                    .take(d)
                    .collect::<Vec<_>>()
            })
            .collect();

        let (bv, nbits) = Self::compress_into_bitvector(
            docs_search_results
                .into_iter()
                .flat_map(|r| r.into_iter())
                .map(|(_distance, doc_id)| doc_id as u64),
            n_vecs,
            d,
        );

        Self {
            n_vecs,
            d,
            //neighbours: neighbours.into_boxed_slice(),
            neighbours: bv,
            nbits,
        }
    }

    pub fn new_from_serialized(path: &str, limit: Option<usize>) -> Self {
        println!("Reading KNN from file: {:}", path);
        let serialized: Vec<u8> = fs::read(path).unwrap();
        let knn = bincode::deserialize::<Knn>(&serialized).unwrap();

        println!("Number of vectors: {:}", knn.n_vecs);
        println!("Number of neighbors in the file: {:}", knn.d);

        let nknn = limit.unwrap_or(knn.d);

        assert!(nknn <= knn.d,
            "The number of neighbors to include for each vector of the dataset can't be greater than the number of neighbours in the precomputed knn file.");

        if nknn == knn.d {
            return knn;
        } else {
            println!("We only take {:} neighbors per element!", nknn);
        }

        let mut neighbours = BitVectorMut::with_capacity(knn.n_vecs * knn.nbits * nknn);

        for id in 0..knn.n_vecs {
            for i in 0..nknn {
                let bit_offset = id * knn.d * knn.nbits + i * knn.nbits;
                let neighbor = knn.neighbours.get_bits(bit_offset, knn.nbits).unwrap();
                neighbours.append_bits(neighbor, knn.nbits);
            }
        }

        Knn {
            n_vecs: knn.n_vecs,
            d: nknn,
            neighbours: BitVector::from(neighbours),
            nbits: knn.nbits,
        }
    }

    pub fn serialize(&self, output_file: &str) -> IoResult<()> {
        let serialized = bincode::serialize(self).unwrap();
        let path = output_file.to_string() + ".knn.seismic";
        println!("Saving ... {}", path);
        fs::write(path, serialized)
    }

    #[must_use]
    fn compress_into_bitvector(
        data: impl Iterator<Item = u64>,
        n_vecs: usize,
        d: usize,
    ) -> (BitVector, usize) {
        let nbits = (n_vecs as f32).log2().ceil() as usize;
        let mut neighbours = BitVectorMut::with_capacity(n_vecs * d * nbits);
        for x in data {
            //neighbours.push(x as u32);
            neighbours.append_bits(x, nbits);
        }
        (BitVector::from(neighbours), nbits)
    }

    #[inline]
    pub fn refine<T>(
        &self,
        query: &[f32],
        heap: &mut HeapFaiss,
        visited: &mut HashSet<usize>,
        forward_index: &SparseDataset<T>,
        in_n_knn: usize,
    ) where
        T: DataType,
    {
        //let n_knn = cmp::max(self.d, in_n_knn);
        let n_knn = cmp::min(self.d, in_n_knn);

        let neighbours: Vec<_> = heap.topk();
        for &(_distance, offset) in neighbours.iter() {
            let id = forward_index.offset_to_id(offset);

            for i in 0..n_knn {
                let bit_offset = id * self.d * self.nbits + i * self.nbits;
                let neighbour = self.neighbours.get_bits(bit_offset, self.nbits).unwrap();

                let offset = forward_index.vector_offset(neighbour as usize);

                let len = forward_index.vector_len(neighbour as usize);

                if !visited.contains(&offset) {
                    let (v_components, v_values) = forward_index.get_with_offset(offset, len);
                    let distance = dot_product_dense_sparse(query, v_components, v_values);
                    visited.insert(offset);
                    heap.push_with_id(-1.0 * distance, offset);
                }
            }
        }
    }
}
