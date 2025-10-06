use core::hash::Hash;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
    fs::File,
    hint::assert_unchecked,
    io::{BufReader, BufWriter},
};
//use std::time::Instant;

use itertools::Itertools;
use rand::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{ComponentType, SparseDataset, ValueType};

pub fn read_from_path<D: DeserializeOwned>(path: &str) -> Result<D, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);
    // let config = bincode::config::standard();
    let config = bincode::config::standard()
        .with_fixed_int_encoding()  
        .with_little_endian();
    let result = bincode::serde::decode_from_std_read::<D, _, _>(&mut file, config)?;
    Ok(result)
}

pub fn write_to_path<E: Serialize>(val: E, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    //let config = bincode::config::standard();
    let config = bincode::config::standard()
        .with_fixed_int_encoding()  
        .with_little_endian();
    bincode::serde::encode_into_std_write(val, &mut file, config)?;
    Ok(())
}

/// A min-heap that stores the top k elements
#[derive(Clone)]
pub struct KHeap<T> {
    bh: BinaryHeap<Reverse<T>>,
    k: usize,
}

impl<T: Ord> KHeap<T> {
    #[inline]
    pub fn new(k: usize) -> Self {
        assert!(k > 0);
        Self {
            bh: BinaryHeap::with_capacity(k),
            k,
        }
    }

    #[inline]
    pub fn push(&mut self, item: T) {
        unsafe { assert_unchecked(self.k > 0 && self.bh.capacity() == self.k) };
        if self.bh.len() < self.k {
            self.bh.push(Reverse(item));
        } else {
            let mut min = self.bh.peek_mut().unwrap();
            if item > min.0 {
                *min = Reverse(item);
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bh.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn peek(&self) -> &T {
        &self.bh.peek().unwrap().0
    }

    #[inline]
    pub fn into_sorted_vec(self) -> Vec<T> {
        // Zero-cost abstraction
        self.bh.into_sorted_vec().into_iter().map(|i| i.0).collect()
    }
}

#[derive(Clone, PartialEq)]
pub struct ScoredItem {
    pub id: usize,
    pub score: f32,
}

impl ScoredItem {
    pub fn new(id: usize, score: f32) -> Self {
        Self { id, score }
    }
}

impl Eq for ScoredItem {}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        unsafe { self.score.partial_cmp(&other.score).unwrap_unchecked() }
    }
}

/// Instead of string doc_ids we store their offsets in the forward_index and the lengths of the vectors
/// This allows us to save the random accesses that would be needed to access exactly these values from the
/// forward index. The values of each doc are packed into a single u64 in `packed_postings`.
/// We use 48 bits for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48.
/// We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
/// Preferably use #[repr(packed)], if u48 becomes a thing: https://github.com/rust-lang/rfcs/issues/2903
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PackedPostingBlock {
    n: u64,
}

impl PackedPostingBlock {
    #[inline]
    pub fn new_pack(offset: u64, len: u16) -> Self {
        Self {
            n: ((offset) << 16) | (len as u64),
        }
    }

    #[inline]
    pub fn unpack(&self) -> (usize, usize) {
        (
            (self.n >> 16) as usize,
            (self.n & (u16::MAX as u64)) as usize,
        )
    }
}

/// Computes the size of the intersection of two unsorted lists of integers.
pub fn intersection<T: Eq + Hash + Clone>(s: &[T], groundtruth: &[T]) -> usize {
    let s_set: HashSet<_> = s.iter().cloned().collect();
    let mut size = 0;
    for v in groundtruth {
        if s_set.contains(v) {
            size += 1;
        }
    }
    size
}

const THRESHOLD_BINARY_SEARCH: usize = 10;

#[inline]
#[must_use]
pub fn conditionally_densify<C>(
    query_components: &[C],
    query_values: &[f32],
    query_dim: usize,
) -> Option<Vec<f32>>
where
    C: ComponentType,
{
    if query_components.len() > THRESHOLD_BINARY_SEARCH && query_dim <= 2_usize.pow(18) {
        let mut vec = vec![0.0; query_dim];
        for (&i, &v) in query_components.iter().zip(query_values) {
            vec[i.as_()] = v;
        }
        Some(vec)
    } else {
        None
    }
}

#[inline]
pub fn prefetch_read_slice<T>(data: &[T]) {
    let ptr = data.as_ptr() as *const i8;
    // Cache line size on x86 is 64 bytes.
    // The function is written with a pointer because iterating the array seems to prevent loop unrolling, for some reason.
    for i in (0..size_of_val(data)).step_by(64) {
        core::intrinsics::prefetch_read_data::<_, 0>(ptr.wrapping_add(i));
    }
}

fn compute_centroid_assignments_approx_dot_product<C: ComponentType, T: ValueType>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(usize, T)>],
    dataset: &SparseDataset<C, T>,
    centroids_doc_ids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)> {
    let mut scores = vec![0_f32; centroids_doc_ids.len()];

    doc_ids
        .iter()
        .map(|&doc_id| {
            scores.iter_mut().for_each(|v| *v = 0_f32);
            for (&component_id, &value) in dataset
                .iter_vector(doc_id)
                .k_largest_by(doc_cut, |a, b| a.1.partial_cmp(b.1).unwrap())
            {
                for &(centroid_id, score) in inverted_index[component_id.as_()].iter() {
                    scores[centroid_id] += score.to_f32().unwrap() * value.to_f32().unwrap();
                }
            }

            let (&max_centroid_doc_id, _) = centroids_doc_ids
                .iter()
                .zip(scores.iter())
                .filter(|(centroid_doc_id, _)| !to_avoid.contains(centroid_doc_id))
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
/// The paramenter `doc_cut` specifies how many components of the document vector to consider whiel computing the dot product.
pub fn do_random_kmeans_on_docids_ii_approx_dot_product<C, T>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<C, T>,
    min_cluster_size: usize,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    C: ComponentType,
    T: ValueType,
{
    let seed = 1142;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_doc_ids: Vec<_> = doc_ids
        .choose_multiple(&mut rng, n_clusters)
        .copied()
        .collect();

    // Build an inverted index for the centroids
    let mut inverted_index = vec![Vec::new(); dataset.dim()];

    for (centroid_id, &centroid_doc_id) in centroid_doc_ids.iter().enumerate() {
        for (&c, &score) in dataset.iter_vector(centroid_doc_id) {
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

fn compute_centroid_assignments_dot_product<C, T>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(T, usize)>],
    dataset: &SparseDataset<C, T>,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    C: ComponentType,
    T: ValueType,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        // Densify the vector
        let mut dense_vector: Vec<T> = vec![T::zero(); dataset.dim()];
        for (&c, &v) in dataset.iter_vector(doc_id) {
            dense_vector[c.as_()] = v;
        }

        let doc_components = dataset.get(doc_id).0;
        //FIXME: avoiding this copy requires to parameterize the dot_product computation w.r.t. to the
        // values type. Not sure if this is worth it.
        let doc_values = dataset
            .get(doc_id)
            .1
            .iter()
            .map(|v| v.to_f32().unwrap())
            .collect::<Vec<_>>();

        let dense_vector = conditionally_densify(doc_components, &doc_values, dataset.dim());

        let mut visited = to_avoid.clone();

        // Sort query terms by score and evaluate the posting list only for the top ones
        let (max_centroid_id, _dot) = dataset
            .iter_vector(doc_id)
            .k_largest_by(doc_cut, |a, b| a.1.partial_cmp(b.1).unwrap())
            .flat_map(|(&component_id, &_value)| inverted_index[component_id.as_()].iter())
            .filter(|&(_score, centroid_id)| visited.insert(*centroid_id))
            .map(|&(_score, centroid_id)| {
                let (v_components, v_values) = dataset.get(centroid_id);
                let dot = C::compute_dot_product(
                    dense_vector.as_deref(),
                    doc_components,
                    &doc_values,
                    v_components,
                    v_values,
                );
                (centroid_id, dot)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((centroids[0], 0.0));

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
pub fn do_random_kmeans_on_docids_ii_dot_product<C, T>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<C, T>,
    min_cluster_size: usize,
    pruning_factor: f32,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    C: ComponentType,
    T: ValueType,
{
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let pruned_list_size = 5.max((doc_ids.len() as f32 * pruning_factor) as usize);

    // Build an inverted index for the centroids
    let mut inverted_index = vec![Vec::new(); dataset.dim()];

    for &centroid_id in centroid_ids.iter() {
        for (&c, &score) in dataset.iter_vector(centroid_id) {
            inverted_index[c.as_()].push((score, centroid_id));
        }
    }

    let inverted_index = inverted_index
        .into_iter()
        .map(|list| {
            list.into_iter()
                .k_largest_by(pruned_list_size, |a, b| a.0.partial_cmp(&b.0).unwrap())
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

fn compute_centroid_assignments<C: ComponentType, T: ValueType>(
    doc_ids: &[usize],
    dataset: &SparseDataset<C, T>,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
) -> Vec<(usize, usize)> {
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        let doc_components = dataset.get(doc_id).0;
        // FIXME: avoiding this copy requires to parameterize the dot_product computation w.r.t. to the
        // values type. Not sure if this is worth it.
        let doc_values = dataset
            .get(doc_id)
            .1
            .iter()
            .map(|v| v.to_f32().unwrap())
            .collect::<Vec<_>>();

        let dense_vector = conditionally_densify(doc_components, &doc_values, dataset.dim());

        let (centroid_max, _dot) = centroids
            .iter()
            .map(|&centroid_id| {
                let (v_components, v_values) = dataset.get(centroid_id);
                let dot = C::compute_dot_product(
                    dense_vector.as_deref(),
                    doc_components,
                    &doc_values,
                    v_components,
                    v_values,
                );
                (centroid_id, dot)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            // The cluster(s) may be small... and also the only one(s).
            .unwrap_or((centroids[0], 0.0));

        centroid_assignments.push((centroid_max, doc_id));
    }

    centroid_assignments
}

pub fn do_random_kmeans_on_docids<C: ComponentType, T: ValueType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<C, T>,
    min_cluster_size: usize,
) -> Vec<(usize, usize)> {
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
