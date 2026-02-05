use std::{
    collections::{BinaryHeap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
};

use itertools::Itertools;
use num_traits::{AsPrimitive, ToPrimitive};
use rand::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

use crate::index_traits::{EncoderFor, SeismicBuildDataset};
use vectorium::{
    ComponentType, Distance, DotProduct, QueryEvaluator, SparseVectorView, ValueType, VectorEncoder,
};

/// Read a bincode-serialized value from `path` using fixed-int, little-endian encoding.
pub fn read_from_path<D: DeserializeOwned>(path: &str) -> Result<D, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);
    // let config = bincode::config::standard();
    let config = bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian();
    let result = bincode::serde::decode_from_std_read::<D, _, _>(&mut file, config)?;
    Ok(result)
}

/// Write a value to `path` using bincode with fixed-int, little-endian encoding.
pub fn write_to_path<E: Serialize>(val: E, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    //let config = bincode::config::standard();
    let config = bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian();
    bincode::serde::encode_into_std_write(val, &mut file, config)?;
    Ok(())
}

/// A max-heap that keeps the k smallest elements seen so far.
#[derive(Clone)]
pub struct KHeap<T> {
    bh: BinaryHeap<T>,
    k: usize,
}

impl<T: Ord> KHeap<T> {
    #[inline]
    /// Create a heap that retains the k smallest elements; panics if `k == 0`.
    pub fn new(k: usize) -> Self {
        assert!(k > 0);
        Self {
            bh: BinaryHeap::with_capacity(k),
            k,
        }
    }

    #[inline]
    /// Insert an item, keeping only the k smallest elements.
    pub fn push(&mut self, item: T) {
        if self.bh.len() < self.k {
            self.bh.push(item);
        } else {
            let mut max = self.bh.peek_mut().unwrap();
            if item < *max {
                *max = item;
            }
        }
    }

    #[inline]
    /// Return the number of elements currently stored.
    pub fn len(&self) -> usize {
        self.bh.len()
    }

    #[inline]
    /// Return whether the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    /// Return the current maximum among the retained elements; panics if empty.
    pub fn peek(&self) -> &T {
        self.bh.peek().unwrap()
    }

    #[inline]
    /// Consume the heap and return the retained elements in ascending order.
    pub fn into_sorted_vec(self) -> Vec<T> {
        self.bh.into_sorted_vec()
    }
}

pub(crate) fn quantize<T: ValueType>(values: &[T]) -> (f32, f32, Vec<u8>) {
    assert!(!values.is_empty());

    const MAX_QUANT: f32 = u8::MAX as f32;

    // Compute min and max values in the vector
    let (min, max) = values
        .iter()
        .map(|v| v.to_f32().unwrap())
        .minmax_by(|a, b| a.total_cmp(b))
        .into_option()
        .unwrap();

    // Quantization splits the range [min, max] into n_classes blocks of equal size (max-min)/n_clasess.
    // (Exponential quantization could be possible as well.)
    let quant = (max - min) / MAX_QUANT;
    let quantized_values = values
        .iter()
        .map(|&v| ((v.to_f32().unwrap() - min) / quant).round() as u8)
        .collect();

    (min, quant, quantized_values)
}

fn iter_components_values<'a, C, V>(
    vector: &'a SparseVectorView<'a, C, V>,
) -> impl Iterator<Item = (C, V)> + 'a
where
    C: ComponentType,
    V: ValueType,
{
    vector
        .components()
        .iter()
        .copied()
        .zip(vector.values().iter().copied())
}

fn compute_centroid_assignments_approx_dot_product<S, T>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(usize, T)>],
    dataset: &S,
    centroids_doc_ids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
    T: ValueType,
{
    let mut scores = vec![0_f32; centroids_doc_ids.len()];

    doc_ids
        .iter()
        .map(|&doc_id| {
            scores.iter_mut().for_each(|v| *v = 0_f32);
            let posting = dataset.get(doc_id as u64);
            let iter = iter_components_values(&posting).k_largest_by(doc_cut, |a, b| {
                a.1.to_f32().unwrap().total_cmp(&b.1.to_f32().unwrap())
            });
            for (component_id, value) in iter {
                for &(centroid_id, score) in inverted_index[component_id.as_()].iter() {
                    scores[centroid_id] += score.to_f32().unwrap() * value.to_f32().unwrap();
                }
            }

            let (&max_centroid_doc_id, _) = centroids_doc_ids
                .iter()
                .zip(scores.iter())
                .filter(|(centroid_doc_id, _)| !to_avoid.contains(centroid_doc_id))
                .max_by(|a, b| a.1.to_f32().unwrap().total_cmp(&b.1.to_f32().unwrap()))
                .unwrap_or((&centroids_doc_ids[0], &0.0));

            (max_centroid_doc_id, doc_id)
        })
        .collect()
}

/// Perform a random k-means clustering on a set of document ids.
/// The function returns a vector of pairs (cluster_id, doc_id) where cluster_id is the id of the cluster to which the document belongs.
/// The vector is sorted by cluster_id.
///
/// The function uses a simple pruned inverted index to speed up the computation and computes the
/// true dot product between the document and the centroids.
/// The parameter `doc_cut` specifies how many components of the document vector to consider while computing the dot product.
pub(crate) fn do_random_kmeans_on_docids_ii_approx_dot_product<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let seed = 1142;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_doc_ids: Vec<_> = doc_ids
        .choose_multiple(&mut rng, n_clusters)
        .copied()
        .collect();

    // Build an inverted index for the centroids
    let mut inverted_index = vec![Vec::new(); dataset.input_dim()];

    for (centroid_id, &centroid_doc_id) in centroid_doc_ids.iter().enumerate() {
        let posting = dataset.get(centroid_doc_id as u64);
        for (c, score) in iter_components_values(&posting) {
            inverted_index[c.as_()].push((centroid_id, score));
        }
    }

    let mut centroid_assignments = compute_centroid_assignments_approx_dot_product(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_doc_ids,
        &HashSet::new(),
        doc_cut,
    );

    // Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assignments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assignments.sort_unstable();

    for group in centroid_assignments.chunk_by(
        // group by centroid_id
        |&(centroid_doc_id_a, _doc_id_a), &(centroid_doc_id_b, _doc_id_b)| {
            centroid_doc_id_a == centroid_doc_id_b
        },
    ) {
        let centroid_doc_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_doc_id);
        } else {
            final_assignments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assignments = compute_centroid_assignments_approx_dot_product(
        to_be_reassigned.as_slice(),
        &inverted_index,
        dataset,
        &centroid_doc_ids,
        &removed_centroids,
        doc_cut,
    );

    final_assignments.extend(centroid_assignments);

    assert_eq!(
        final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assignments.sort();

    final_assignments
}

fn compute_centroid_assignments_dot_product<A, T, S>(
    doc_ids: &[usize],
    inverted_index: &[A],
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    A: AsRef<[(T, usize)]>,
    T: ValueType,
    S: SeismicBuildDataset,
    for<'a> <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>: Send,
    for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>:
        QueryEvaluator<<EncoderFor<S> as VectorEncoder>::EncodedVector<'a>, Distance = DotProduct>,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        let doc_vec = dataset.get(doc_id as u64);
        let max_centroid_id = {
            let evaluator = dataset.encoder().vector_evaluator(doc_vec);
            let mut visited = to_avoid.clone();

            // Sort query terms by score and evaluate the posting list only for the top ones
            let components = doc_vec.components();
            let values = doc_vec.values();
            let top_components: Vec<_> = components
                .iter()
                .copied()
                .zip(values.iter().copied())
                .k_largest_by(doc_cut, |a, b| {
                    a.1.to_f32().unwrap().total_cmp(&b.1.to_f32().unwrap())
                })
                .map(|(component_id, _value)| component_id)
                .collect();

            let mut max_centroid_id = centroids[0];
            let mut max_dot = 0.0_f32;
            for component_id in top_components {
                for &(_score, centroid_id) in inverted_index[component_id.as_()].as_ref().iter() {
                    if !visited.insert(centroid_id) {
                        continue;
                    }
                    let centroid_vec = dataset.get(centroid_id as u64);
                    let dot = evaluator.compute_distance(centroid_vec).distance();
                    if dot > max_dot {
                        max_dot = dot;
                        max_centroid_id = centroid_id;
                    }
                }
            }

            max_centroid_id
        };

        centroid_assignments.push((max_centroid_id, doc_id));
    }

    centroid_assignments
}

/// Perform a random k-means clustering on a set of document ids.
/// The function returns a vector of pairs (cluster_id, doc_id) where cluster_id is the id of the cluster to which the document belongs.
/// The vector is sorted by cluster_id.
///
/// The function uses a simple pruned inverted index to speed up the computation and computes the
/// true dot product between the document and the centroids.
/// The parameter `pruning_factor` controls the size of the pruned inverted index.
/// The parameter `doc_cut` specifies how many components of the document vector to consider while computing the dot product.
pub(crate) fn do_random_kmeans_on_docids_ii_dot_product<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    pruning_factor: f32,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let pruned_list_size = 5.max((doc_ids.len() as f32 * pruning_factor) as usize);

    // Build an inverted index for the centroids
    let mut inverted_index = vec![Vec::new(); dataset.input_dim()];

    for &centroid_id in centroid_ids.iter() {
        let posting = dataset.get(centroid_id as u64);
        for (c, score) in iter_components_values(&posting) {
            inverted_index[c.as_()].push((score, centroid_id));
        }
    }

    let inverted_index = inverted_index
        .into_iter()
        .map(|list| {
            list.into_iter()
                .k_largest_by(pruned_list_size, |a, b| {
                    a.0.to_f32().unwrap().total_cmp(&b.0.to_f32().unwrap())
                })
                .collect_vec()
        })
        .collect_vec();

    let mut centroid_assignments = compute_centroid_assignments_dot_product(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_ids,
        &HashSet::new(),
        doc_cut,
    );

    // Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assignments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assignments.sort_unstable();

    for group in centroid_assignments.chunk_by(
        // group by centroid_id
        |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| centroid_id_a == centroid_id_b,
    ) {
        let centroid_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            final_assignments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assignments = compute_centroid_assignments_dot_product(
        to_be_reassigned.as_slice(),
        &inverted_index,
        dataset,
        &centroid_ids,
        &removed_centroids,
        doc_cut,
    );

    final_assignments.extend(centroid_assignments);

    assert_eq!(
        final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assignments.sort_unstable();

    final_assignments
}

fn compute_centroid_assignments<S>(
    doc_ids: &[usize],
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());
    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }
        let doc_vec = dataset.get(doc_id as u64);
        let centroid_max = {
            let evaluator = dataset.encoder().vector_evaluator(doc_vec);

            let mut centroid_max = centroids[0];
            let mut max_dot = 0.0_f32;
            for &centroid_id in centroids {
                let centroid_vec = dataset.get(centroid_id as u64);
                let dot = evaluator.compute_distance(centroid_vec).distance();
                if dot > max_dot {
                    max_dot = dot;
                    centroid_max = centroid_id;
                }
            }

            centroid_max
        };

        centroid_assignments.push((centroid_max, doc_id));
    }

    centroid_assignments
}

/// Run randomized k-means on document ids using exact dot products for assignment.
/// Returns `(cluster_id, doc_id)` pairs sorted by `cluster_id`.
pub(crate) fn do_random_kmeans_on_docids<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let seed = 42; // You can use any u64 value as the seed
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let mut centroid_assignments =
        compute_centroid_assignments(doc_ids, dataset, &centroid_ids, &HashSet::new());

    // Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assignments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assignments.sort_unstable();

    for group in centroid_assignments.chunk_by(
        // group by centroid_id
        |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| centroid_id_a == centroid_id_b,
    ) {
        let centroid_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            final_assignments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assignments = compute_centroid_assignments(
        to_be_reassigned.as_slice(),
        dataset,
        &centroid_ids,
        &removed_centroids,
    );

    final_assignments.extend(&centroid_assignments);

    assert_eq!(
        final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assignments.sort();

    final_assignments
}
