use core::hash::Hash;
use std::collections::HashSet;

#[allow(unused_imports)]
use rand::{rngs::StdRng, seq::IteratorRandom, thread_rng, SeedableRng};

use crate::{distances::dot_product_dense_sparse, DataType, SparseDataset};

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

#[allow(non_snake_case)]
#[inline]
pub fn prefetch_read_NTA<T>(data: &[T], offset: usize) {
    let _p = data.as_ptr().wrapping_add(offset) as *const i8;

    #[cfg(all(feature = "prefetch", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_prefetch, _MM_HINT_NTA};

        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};

        unsafe {
            _mm_prefetch(_p, _MM_HINT_NTA);
        }
    }

    #[cfg(all(feature = "prefetch", target_arch = "aarch64"))]
    {
        use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY0, _PREFETCH_READ};

        unsafe {
            _prefetch(_p, _PREFETCH_READ, _PREFETCH_LOCALITY0);
        }
    }
}

/// Returns the type name of its argument.
pub fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}

#[inline]
#[must_use]
pub fn binary_search_branchless(data: &[u16], target: u16) -> usize {
    let mut base = 0;
    let mut size = data.len();
    while size > 1 {
        let mid = size / 2;
        let cmp = *unsafe { data.get_unchecked(base + mid - 1) } < target;
        base += if cmp { mid } else { 0 };
        size -= mid;
    }

    base
}

use itertools::Itertools;

fn compute_centroid_assignments_approx_dot_product<T: DataType>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(usize, T)>],
    dataset: &SparseDataset<T>,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)> {
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());
    let mut scores = vec![0_f32; centroids.len()];

    for &doc_id in doc_ids.iter() {
        scores.iter_mut().for_each(|v| *v = 0_f32);
        for (&component_id, &value) in dataset
            .iter_vector(doc_id)
            .sorted_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap())
            .take(doc_cut)
        {
            for &(centroid_id, score) in inverted_index[component_id as usize].iter() {
                scores[centroid_id] += score.to_f32().unwrap() * value.to_f32().unwrap();
            }
        }

        let mut max = 0_f32;
        let mut max_centroid_id = centroids[0];

        for (centroid_id, &score) in scores.iter().enumerate() {
            if score > max && !to_avoid.contains(&centroid_id) {
                max = score;
                max_centroid_id = centroid_id;
            }
        }

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
/// The paramenter `doc_cut` specifies how many components of the document vector to consider whiel computing the dot product.
pub fn do_random_kmeans_on_docids_ii_approx_dot_product<T: DataType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<T>,
    min_cluster_size: usize,
    doc_cut: usize,
) -> Vec<(usize, usize)> {
    let seed = 1142;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    // Build an inverted index for the centroids
    let mut inverted_index = Vec::with_capacity(dataset.dim());
    for _ in 0..dataset.dim() {
        inverted_index.push(Vec::new());
    }

    for (i, &centroid_id) in centroid_ids.iter().enumerate() {
        for (&c, &score) in dataset.iter_vector(centroid_id) {
            inverted_index[c as usize].push((i, score));
        }
    }

    let mut centroid_assigments = compute_centroid_assignments_approx_dot_product(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_ids,
        &HashSet::new(),
        doc_cut,
    );

    // Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assigments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assigments.sort_unstable();

    for group in centroid_assigments.chunk_by(
        // group by centroid_id
        |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| centroid_id_a == centroid_id_b,
    ) {
        let centroid_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            final_assigments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assigments = compute_centroid_assignments_approx_dot_product(
        to_be_reassigned.as_slice(),
        &inverted_index,
        dataset,
        &centroid_ids,
        &removed_centroids,
        doc_cut,
    );

    final_assigments.extend(centroid_assigments);

    assert_eq!(
        final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assigments.sort();

    final_assigments
}

fn compute_centroid_assignments_dot_product<T: DataType>(
    doc_ids: &[usize],
    inverted_index: &[Vec<(T, usize)>],
    dataset: &SparseDataset<T>,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)> {
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
            dense_vector[c as usize] = v;
        }

        let mut max = 0_f32;
        let mut max_centroid_id = centroids[0];

        let mut visited = to_avoid.clone();

        // Sort query terms by score and evaluate the posting list only for the top ones
        for (&component_id, &_value) in dataset
            .iter_vector(doc_id)
            .sorted_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap())
            .take(doc_cut)
        {
            for &(_score, centroid_id) in inverted_index[component_id as usize].iter() {
                if visited.contains(&centroid_id) {
                    continue;
                }
                visited.insert(centroid_id);

                let (v_components, v_values) = dataset.get(centroid_id);
                let dot = dot_product_dense_sparse(&dense_vector, v_components, v_values);
                if dot > max {
                    max = dot;
                    max_centroid_id = centroid_id;
                }
            }
        }

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
/// The paramenter `pruning_factor` controls the size of the pruned inverted index.
/// The parameter `doc_cut` specifies how many components of the document vector to consider while computing the dot product.
pub fn do_random_kmeans_on_docids_ii_dot_product<T: DataType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<T>,
    min_cluster_size: usize,
    pruning_factor: f32,
    doc_cut: usize,
) -> Vec<(usize, usize)> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let pruned_list_size = 5.max((doc_ids.len() as f32 * pruning_factor) as usize);

    // Build an inverted index for the centroids
    let mut inverted_index = Vec::with_capacity(dataset.dim());
    for _ in 0..dataset.dim() {
        inverted_index.push(Vec::new());
    }

    for &centroid_id in centroid_ids.iter() {
        for (&c, &score) in dataset.iter_vector(centroid_id) {
            inverted_index[c as usize].push((score, centroid_id));
        }
    }

    for list in inverted_index.iter_mut() {
        list.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        list.truncate(pruned_list_size);
    }

    let mut centroid_assigments = compute_centroid_assignments_dot_product(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_ids,
        &HashSet::new(),
        doc_cut,
    );

    // Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assigments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assigments.sort_unstable();

    for group in centroid_assigments.chunk_by(
        // group by centroid_id
        |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| centroid_id_a == centroid_id_b,
    ) {
        let centroid_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            final_assigments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assigments = compute_centroid_assignments_dot_product(
        to_be_reassigned.as_slice(),
        &inverted_index,
        dataset,
        &centroid_ids,
        &removed_centroids,
        doc_cut,
    );

    final_assigments.extend(centroid_assigments);

    assert_eq!(
        final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assigments.sort();

    final_assigments
}

fn compute_centroid_assignments<T: DataType>(
    doc_ids: &[usize],
    dataset: &SparseDataset<T>,
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

        let mut dense_vector: Vec<T> = vec![T::zero(); dataset.dim()];
        for (&i, &v) in dataset.iter_vector(doc_id) {
            dense_vector[i as usize] = v;
        }

        let mut centroid_max = centroids[0];
        let mut max = 0_f32;
        for &centroid_id in centroids.iter() {
            let (v_components, v_values) = dataset.get(centroid_id);
            let dot = dot_product_dense_sparse(&dense_vector, v_components, v_values);
            if dot > max {
                max = dot;
                centroid_max = centroid_id;
            }
        }
        centroid_assignments.push((centroid_max, doc_id));
    }

    centroid_assignments
}

pub fn do_random_kmeans_on_docids<T: DataType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<T>,
    min_cluster_size: usize,
) -> Vec<(usize, usize)> {
    let seed = 42; // You can use any u64 value as the seed
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let mut centroid_assigments =
        compute_centroid_assignments(doc_ids, dataset, &centroid_ids, &HashSet::new());

    // Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assigments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assigments.sort_unstable();

    for group in centroid_assigments.chunk_by(
        // group by centroid_id
        |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| centroid_id_a == centroid_id_b,
    ) {
        let centroid_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            final_assigments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assigments = compute_centroid_assignments(
        to_be_reassigned.as_slice(),
        dataset,
        &centroid_ids,
        &removed_centroids,
    );

    final_assigments.extend(&centroid_assigments);

    assert_eq!(
        final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assigments.sort();

    final_assigments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_search() {
        let data = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(binary_search_branchless(&data, v), i);
        }

        assert_eq!(binary_search_branchless(&data, 198), data.len() - 1);
    }
}
