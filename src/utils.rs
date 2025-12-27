use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
    fs::File,
    hash::Hash,
    hint::assert_unchecked,
    io::{BufReader, BufWriter},
};
//use std::time::Instant;

use itertools::Itertools;
use rand::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::*;
use vectorium::{
    Dataset, Distance as VDistance, QueryEvaluator, SparseQuantizer, SparseVector1D, Vector1D,
    VectorEncoder,
};

type ComponentFor<E> = <E as VectorEncoder>::OutputComponentType;
type ValueFor<E> = <E as VectorEncoder>::OutputValueType;
type QueryValueFor<E> = <E as VectorEncoder>::QueryValueType;

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
pub struct ScoredItem<T, I> {
    pub id: I,
    pub score: T,
}

impl<T, I> ScoredItem<T, I> {
    pub fn new(id: I, score: T) -> Self {
        Self { id, score }
    }
}

impl<T, I> Eq for ScoredItem<T, I>
where
    T: PartialEq,
    I: PartialEq,
{
}

impl<T, I> PartialOrd for ScoredItem<T, I>
where
    T: PartialOrd,
    I: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, I> Ord for ScoredItem<T, I>
where
    T: PartialOrd,
    I: PartialEq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        unsafe { self.score.partial_cmp(&other.score).unwrap_unchecked() }
    }
}

/// Instead of storing doc_ids we store their offsets in the forward_index and the lengths of the vectors
/// This allows us to save the random accesses that would be needed to access exactly these values from the
/// forward index. The values of each doc are packed into a single u64 in `packed_postings`.
/// We use 48 bits for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48.
/// We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
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
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

pub(crate) fn quantize<T: ValueType>(values: &[T]) -> (f32, f32, Vec<u8>) {
    assert!(!values.is_empty());

    const MAX_QUANT: f32 = u8::MAX as f32;

    // Compute min and max values in the vector
    let (min, max) = values
        .iter()
        .minmax_by(|a, b| a.partial_cmp(b).unwrap())
        .into_option()
        .unwrap();

    let (min, max) = (min.to_f32().unwrap(), max.to_f32().unwrap());

    // Quantization splits the range [min, max] into n_classes blocks of equal size (max-min)/n_clasess.
    // (Exponential quantization could be possible as well.)
    let quant = (max - min) / MAX_QUANT;
    let quantized_values = values
        .iter()
        .map(|&v| ((v.to_f32().unwrap() - min) / quant).round() as u8)
        .collect();

    (min, quant, quantized_values)
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
pub fn prefetch_read<T>(ptr: *const T) {
    let ptr = ptr as *const u8;

    // Prefetching is much better when done by a constant value, as doing branches to dynamically choose how much to prefetch defeats the point.
    // The milion dollar question: how much to prefetch? It depends on the computer.
    // On newer computers, the optimal amount of data to prefetch is 1, which is why it's the default.
    // Older computers however gain to most advantage by prefetching multiple times.
    const LEN: usize = match envparse::parse_env!(try "PREFETCH_LEN" as usize) {
        Some(l) => l,
        None => 1,
    };
    const CACHE_LINE_SIZE: usize = 64;

    for i in 0..LEN {
        core::intrinsics::prefetch_read_data::<_, 0>(ptr.wrapping_add(i * CACHE_LINE_SIZE));
    }
}

fn iter_components_values<'a, V>(
    vector: &'a V,
) -> impl Iterator<Item = (V::ComponentType, V::ValueType)> + 'a
where
    V: Vector1D,
    V::ComponentType: Copy,
    V::ValueType: Copy,
{
    vector
        .components_as_slice()
        .iter()
        .copied()
        .zip(vector.values_as_slice().iter().copied())
}

fn compute_centroid_assignments_approx_dot_product<S, E, T>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(usize, T)>],
    dataset: &S,
    centroids_doc_ids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: Dataset<E>,
    E: SparseQuantizer,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    for<'a> <E as VectorEncoder>::EncodedVector<'a>:
        Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
    T: ValueType,
{
    let mut scores = vec![0_f32; centroids_doc_ids.len()];

    doc_ids
        .iter()
        .map(|&doc_id| {
            scores.iter_mut().for_each(|v| *v = 0_f32);
            let posting = dataset.get(doc_id as u64);
            let iter = iter_components_values(&posting)
                .k_largest_by(doc_cut, |a, b| a.1.partial_cmp(&b.1).unwrap());
            for (component_id, value) in iter {
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
/// The parameter `doc_cut` specifies how many components of the document vector to consider while computing the dot product.
pub fn do_random_kmeans_on_docids_ii_approx_dot_product<S, E>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: Dataset<E>,
    E: SparseQuantizer,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    for<'a> <E as VectorEncoder>::EncodedVector<'a>:
        Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
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

fn compute_centroid_assignments_dot_product<A, T, S, E>(
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
    S: Dataset<E>,
    E: SparseQuantizer,
    E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    QueryValueFor<E>: ValueType,
    E: VectorEncoder<QueryValueType = f32>,
    for<'a> <E as VectorEncoder>::EncodedVector<'a>:
        Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        let doc_vec = dataset.get(doc_id as u64);
        let components = doc_vec.components_as_slice();
        let values = doc_vec.values_as_slice();
        let query_values: Vec<_> = values.iter().map(|v| v.to_f32().unwrap()).collect();
        let query = SparseVector1D::new(components, query_values.as_slice());
        let evaluator = dataset.get_query_evaluator(&query);
        let mut visited = to_avoid.clone();

        // Sort query terms by score and evaluate the posting list only for the top ones
        let (max_centroid_id, _dot) = components
            .iter()
            .copied()
            .zip(values.iter().copied())
            .k_largest_by(doc_cut, |a, b| a.1.partial_cmp(&b.1).unwrap())
            .flat_map(|(component_id, _value)| inverted_index[component_id.as_()].as_ref().iter())
            .filter(|&(_score, centroid_id)| visited.insert(*centroid_id))
            .map(|&(_score, centroid_id)| {
                let centroid_vec = dataset.get(centroid_id as u64);
                let dot = evaluator.compute_distance(centroid_vec).distance();
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
pub fn do_random_kmeans_on_docids_ii_dot_product<S, E>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    pruning_factor: f32,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: Dataset<E>,
    E: SparseQuantizer,
    E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    QueryValueFor<E>: ValueType,
    E: VectorEncoder<QueryValueType = f32>,
    for<'a> <E as VectorEncoder>::EncodedVector<'a>:
        Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
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

fn compute_centroid_assignments<S, E>(
    doc_ids: &[usize],
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
) -> Vec<(usize, usize)>
where
    S: Dataset<E>,
    E: SparseQuantizer,
    E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    QueryValueFor<E>: ValueType,
    E: VectorEncoder<QueryValueType = f32>,
    for<'a> <E as VectorEncoder>::EncodedVector<'a>:
        Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());
    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }
        let doc_vec = dataset.get(doc_id as u64);
        let components = doc_vec.components_as_slice();
        let values = doc_vec.values_as_slice();
        let query_values: Vec<_> = values.iter().map(|v| v.to_f32().unwrap()).collect();
        let query = SparseVector1D::new(components, query_values.as_slice());
        let evaluator = dataset.get_query_evaluator(&query);

        let (centroid_max, _dot) = centroids
            .iter()
            .map(|&centroid_id| {
                let centroid_vec = dataset.get(centroid_id as u64);
                let dot = evaluator.compute_distance(centroid_vec).distance();
                (centroid_id, dot)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            // The cluster(s) may be small... and also the only one(s).
            .unwrap_or((centroids[0], 0.0));

        centroid_assignments.push((centroid_max, doc_id));
    }

    centroid_assignments
}

pub fn do_random_kmeans_on_docids<S, E>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
) -> Vec<(usize, usize)>
where
    S: Dataset<E>,
    E: SparseQuantizer,
    E: VectorEncoder<QueryComponentType = ComponentFor<E>>,
    ComponentFor<E>: ComponentType,
    ValueFor<E>: ValueType,
    QueryValueFor<E>: ValueType,
    E: VectorEncoder<QueryValueType = f32>,
    for<'a> <E as VectorEncoder>::EncodedVector<'a>:
        Vector1D<ComponentType = ComponentFor<E>, ValueType = ValueFor<E>>,
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
