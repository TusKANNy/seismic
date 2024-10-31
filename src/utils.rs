use core::hash::Hash;
use std::collections::{HashMap, HashSet};
//use std::time::Instant;

use rand::{seq::IteratorRandom, thread_rng};

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

fn compute_centroid_assignments<T: DataType>(
    doc_ids: &[usize],
    inverted_index: &[Vec<usize>],
    dataset: &SparseDataset<T>,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
) -> Vec<(usize, usize)> {
    const QUERY_CUT: usize = 10; // Number of current doc terms to evaluate

    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let mut centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        //densify the vector
        let mut dense_vector: Vec<T> = vec![T::zero(); dataset.dim()];
        for (&c, &v) in dataset.iter_vector(doc_id) {
            dense_vector[c as usize] = v;
        }

        let mut max = 0_f32;
        let mut max_centroid_id = centroids[0];

        //println!("Start {} max: {}", doc_id, max);

        let mut visited = to_avoid.clone();

        // Sort query terms by score and evaluate the posting list only for the top ones
        for (&component_id, &_value) in dataset
            .iter_vector(doc_id)
            .sorted_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap())
            .take(QUERY_CUT)
        {
            for &centroid_id in inverted_index[component_id as usize].iter() {
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
            //println!("\tpre max: {} {} doc_id {}", max, max_centroid_id, doc_id);
        }
        // println!("final max: {} {} doc_id {}\n", max, max_centroid_id, doc_id);

        centroid_assignments.push((max_centroid_id, doc_id));
    }

    centroid_assignments
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::BufWriter;

pub fn do_random_kmeans_on_docids_2<T: DataType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<T>,
    min_cluster_size: usize,
) -> Vec<Vec<usize>> {
    // if doc_ids.len() < 5000 {
    //     return Vec::new();
    // }

    // let file = File::create("output.bin").unwrap();
    // let writer = BufWriter::new(file);

    // // Serialize and write to the file using bincode
    // bincode::serialize_into(writer, &doc_ids).expect("Failed to serialize data");

    // let time = Instant::now();
    // Create a seeded RNG
    let seed = 42; // You can use any u64 value as the seed
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    /// Build an inverted index for the centroids
    let mut inverted_index = Vec::with_capacity(dataset.dim());
    for _ in 0..dataset.dim() {
        inverted_index.push(Vec::new());
    }

    for &centroid_id in centroid_ids.iter() {
        for (&c, _score) in dataset.iter_vector(centroid_id) {
            inverted_index[c as usize].push(centroid_id);
        }
    }

    let mut centroid_assigments = compute_centroid_assignments(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_ids,
        &HashSet::new(),
    );

    /// Prune too small clusters and reassign the documents to the closest cluster
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut final_assigments = Vec::with_capacity(doc_ids.len());
    let mut removed_centroids = HashSet::new();

    centroid_assigments.sort_unstable();

    for (centroid_id, chunk) in &centroid_assigments
        .into_iter()
        .group_by(|(centroid_id, _doc_id)| *centroid_id)
    {
        let vec_chunk = chunk.collect::<Vec<_>>();
        if vec_chunk.len() <= min_cluster_size {
            to_be_reassigned.extend(vec_chunk.into_iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            final_assigments.extend(vec_chunk.into_iter());
        }
    }

    //println!("to_be_reassigned: {}", to_be_reassigned.len());

    assert_eq!(
        to_be_reassigned.len() + final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let centroid_assigments = compute_centroid_assignments(
        to_be_reassigned.as_slice(),
        &inverted_index,
        dataset,
        &centroid_ids,
        &removed_centroids,
    );

    final_assigments.extend(centroid_assigments);

    assert_eq!(
        final_assigments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    let hash_map = centroid_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect::<HashMap<_, _>>();

    final_assigments.sort_unstable_by_key(|(centroid_id, _doc_id)| hash_map.get(centroid_id));

    assert_eq!(
        final_assigments
            .iter()
            .map(|(_, d)| d)
            .copied()
            .collect::<HashSet<_>>(),
        doc_ids.iter().copied().collect::<HashSet<_>>(),
    );

    let mut inverted_lists = Vec::new();
    for (_centroid_id, group) in &final_assigments
        .into_iter()
        .group_by(|(centroid_id, _doc_id)| *centroid_id)
    {
        let vec_group = group
            .map(|(_centroid_id, doc_id)| doc_id)
            .collect::<Vec<_>>();
        inverted_lists.push(vec_group);
    }

    inverted_lists
}

pub fn do_random_kmeans_on_docids<T: DataType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<T>,
    min_cluster_size: usize,
) -> Vec<Vec<usize>> {
    // let time = Instant::now();
    let seed = 42; // You can use any u64 value as the seed
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let mut inverted_lists: Vec<Vec<_>> = (0..n_clusters).map(|_| Vec::new()).collect();

    for &doc_id in doc_ids {
        let mut dense_vector: Vec<T> = vec![T::zero(); dataset.dim()];
        //densify the vector

        for (&i, &v) in dataset.iter_vector(doc_id) {
            dense_vector[i as usize] = v;
        }
        let mut argmax = 0;
        let mut max = 0_f32;
        for (i, &c_id) in centroid_ids.iter().enumerate() {
            let (v_components, v_values) = dataset.get(c_id);
            let dot = dot_product_dense_sparse(&dense_vector, v_components, v_values);
            if dot > max {
                max = dot;
                argmax = i;
            }
        }
        inverted_lists[argmax].push(doc_id);
    }

    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters.

    //let mut how_many = 0;
    //let mut total = 0;
    for inverted_list in inverted_lists.iter_mut() {
        // if !inverted_list.is_empty() {
        //     total += 1;
        // }
        if !inverted_list.is_empty() && inverted_list.len() <= min_cluster_size {
            //how_many += 1;
            to_be_reassigned.extend(inverted_list.iter());
            inverted_list.clear();
        }
    }

    println!("to_be_reassigned: {}", to_be_reassigned.len());
    for &doc_id in to_be_reassigned.iter() {
        println!("remapping {}", doc_id);
        let mut dense_vector: Vec<T> = vec![T::zero(); dataset.dim()];

        // Densify the vector
        for (&i, &v) in dataset.iter_vector(doc_id) {
            dense_vector[i as usize] = v;
        }

        let mut argmax = 0;
        let mut max = 0_f32;
        for (i, (il, &c_id)) in inverted_lists.iter().zip(centroid_ids.iter()).enumerate() {
            if il.len() <= min_cluster_size {
                continue;
            }

            // Questo è sbagliato!!!
            let (v_components, v_values) = dataset.get(c_id);

            let dot = dot_product_dense_sparse(&dense_vector, v_components, v_values);

            if dot > max {
                max = dot;
                argmax = i;
            }
        }

        inverted_lists[argmax].push(doc_id);
    }

    // println!("\tto_be_reassigned: {}", to_be_reassigned.len());
    // println!("\thow many: {how_many} out of {total}");

    // println!("Elapsed Time {:}", time.elapsed().as_micros());

    inverted_lists
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
