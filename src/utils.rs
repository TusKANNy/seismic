use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
    fs::File,
    hash::{DefaultHasher, Hasher},
    hint::assert_unchecked,
    io::{BufReader, BufWriter},
    time::Instant,
};
//use std::time::Instant;

use itertools::Itertools;
use metis::Graph;
use rand::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{sparse_dataset::SparseDatasetGeneric, *};

pub fn read_from_path<D: DeserializeOwned>(path: &str) -> Result<D, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);
    let config = bincode::config::standard();
    let result = bincode::serde::decode_from_std_read::<D, _, _>(&mut file, config)?;
    Ok(result)
}

pub fn write_to_path<E: Serialize>(val: E, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    let config = bincode::config::standard();
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
pub struct ScoredItem<T> {
    pub id: usize,
    pub score: T,
}

impl<T> ScoredItem<T> {
    pub fn new(id: usize, score: T) -> Self {
        Self { id, score }
    }
}

impl<T> Eq for ScoredItem<T> where T: PartialEq {}

impl<T> PartialOrd for ScoredItem<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ScoredItem<T>
where
    T: PartialOrd,
{
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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
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

fn compute_centroid_assignments_approx_dot_product<S: SparseDatasetTrait, T: ValueType>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(usize, T)>],
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)> {
    let mut scores = vec![0_f32; centroids.len()];

    doc_ids
        .iter()
        .map(|&doc_id| {
            scores.iter_mut().for_each(|v| *v = 0_f32);
            for (component_id, value) in dataset
                .get_iter(doc_id)
                .k_largest_by(doc_cut, |a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                for &(centroid_id, score) in inverted_index[component_id.as_()].iter() {
                    scores[centroid_id] += score.to_f32().unwrap() * value.to_f32().unwrap();
                }
            }

            let (&max_centroid_id, _) = centroids
                .iter()
                .zip(scores.iter())
                .filter(|(centroid_id, _)| !to_avoid.contains(centroid_id))
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or((&centroids[0], &0.0));

            (max_centroid_id, doc_id)
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
pub fn do_random_kmeans_on_docids_ii_approx_dot_product<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SparseDatasetTrait,
{
    let seed = 1142;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids: Vec<_> = doc_ids
        .choose_multiple(&mut rng, n_clusters)
        .copied()
        .collect();

    // Build an inverted index for the centroids
    let mut inverted_index = vec![Vec::new(); dataset.dim()];

    for (i, &centroid_id) in centroid_ids.iter().enumerate() {
        for (c, score) in dataset.get_iter(centroid_id) {
            inverted_index[c.as_()].push((i, score));
        }
    }

    let mut centroid_assignments = compute_centroid_assignments_approx_dot_product(
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

    let centroid_assignments = compute_centroid_assignments_approx_dot_product(
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
    S: SparseDatasetTrait,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        let prepared_query = dataset.prepare_query(dataset.get_iter(doc_id));
        let mut visited = to_avoid.clone();

        // Sort query terms by score and evaluate the posting list only for the top ones
        let (max_centroid_id, _dot) = dataset
            .get_iter(doc_id)
            .k_largest_by(doc_cut, |a, b| a.1.partial_cmp(&b.1).unwrap())
            .flat_map(|(component_id, _value)| inverted_index[component_id.as_()].as_ref().iter())
            .filter(|&(_score, centroid_id)| visited.insert(*centroid_id))
            .map(|&(_score, centroid_id)| {
                let dot = dataset.dot_product_from_id(&prepared_query, centroid_id);
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
pub fn do_random_kmeans_on_docids_ii_dot_product<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    pruning_factor: f32,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SparseDatasetTrait,
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
        for (c, score) in dataset.get_iter(centroid_id) {
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

fn compute_centroid_assignments<S>(
    doc_ids: &[usize],
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
) -> Vec<(usize, usize)>
where
    S: SparseDatasetTrait,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());
    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }
        let prepared_query = dataset.prepare_query(dataset.get_iter(doc_id));

        let (centroid_max, _dot) = centroids
            .iter()
            .map(|&centroid_id| {
                let dot = dataset.dot_product_from_id(&prepared_query, centroid_id);
                (centroid_id, dot)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            // The cluster(s) may be small... and also the only one(s).
            .unwrap_or((centroids[0], 0.0));

        centroid_assignments.push((centroid_max, doc_id));
    }

    centroid_assignments
}

pub fn do_random_kmeans_on_docids<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
) -> Vec<(usize, usize)>
where
    S: SparseDatasetTrait,
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

/// A symmetrical matrix where the diagonal components are 0. Only the upper part of the matrix is stored.
pub struct HollowSymmetricMatrix<T: Zero + Copy> {
    dim: usize,
    data: Box<[T]>,
}

impl<T: Zero + Copy> HollowSymmetricMatrix<T> {
    pub fn new(dim: usize) -> Self {
        let size = (dim * (dim - 1)) / 2;
        let data = vec![T::zero(); size].into_boxed_slice();
        Self { dim, data }
    }

    /// # Safety
    /// - `j > i`
    /// - `i < self.dim && j < self.dim`
    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        unsafe {
            assert_unchecked(j > i);
            let index = i * self.dim + j - ((i + 2) * (i + 1)) / 2;
            self.data.get_unchecked(index)
        }
    }

    /// # Safety
    /// - `j > i`
    /// - `i < self.dim && j < self.dim`
    pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        unsafe {
            assert_unchecked(j > i);
            let index = i * self.dim + j - ((i + 2) * (i + 1)) / 2;
            self.data.get_unchecked_mut(index)
        }
    }

    /// Iterates the specified row (how it's supposed to be, not how it's represented).
    /// The diagonal element is skipped.
    pub fn iter_row(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let before = (0..i).map(move |j| (j, unsafe { self.get_unchecked(j, i) }));
        let after = ((i + 1)..self.dim).map(move |j| (j, unsafe { self.get_unchecked(i, j) }));
        before.chain(after)
    }
}

#[derive(Serialize, Deserialize)]
pub struct MetisParams {
    pub adjncy: Box<[i32]>,
    pub weights: Box<[i32]>,
    pub xadj: Box<[i32]>,
}

impl MetisParams {
    pub fn build_partitions(&self, n_partitions: i32) -> Vec<i32> {
        print!("\tBuilding partitions ");
        let time = Instant::now();

        let mut part = vec![0; self.xadj.len() - 1];

        Graph::new(1, n_partitions, &self.xadj, &self.adjncy)
            .unwrap()
            .set_adjwgt(&self.weights)
            .part_recursive(part.as_mut_slice())
            .unwrap();

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        part
    }
}

/// Load the dataset's adjacency matrix
///
/// Hash the dataset's components and offsets so that its adjacency can be cached (as it's a very long operation)
pub fn build_or_load_metis_params<C, V, O, AC, AV>(
    dataset: &SparseDatasetGeneric<C, V, O, AC, AV>,
) -> MetisParams
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + SpaceUsage + Hash,
    AC: AsRef<[C]> + SpaceUsage + Hash,
    AV: AsRef<[V]> + SpaceUsage,
{
    let mut s = DefaultHasher::new();
    dataset.components().hash(&mut s);
    dataset.offsets().hash(&mut s);
    let hash = s.finish();
    let filename = format!("cached_adjacency_{}", hash);
    if !std::fs::exists(filename.as_str()).is_ok_and(|b| b) {
        println!("Adjacency matrix not cached. Creating.");
        let params = dataset.adjacency_matrix_metis();

        println!("Saving ... {}", filename);
        write_to_path(&params, filename.as_str()).unwrap();

        params
    } else {
        println!("Loading adjacency matrix {}.", filename.as_str());
        read_from_path(filename.as_str()).unwrap()
    }
}
