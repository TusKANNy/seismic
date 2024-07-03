use core::hash::Hash;
use std::collections::HashSet;
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

pub fn do_random_kmeans_on_docids<T: DataType>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &SparseDataset<T>,
    min_cluster_size: usize,
) -> Vec<Vec<usize>> {
    // let time = Instant::now();
    let mut rng = thread_rng();
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

    let mut to_be_replaced = Vec::new(); // docids that belong to too small clusters.

    //let mut how_many = 0;
    //let mut total = 0;
    for inverted_list in inverted_lists.iter_mut() {
        // if !inverted_list.is_empty() {
        //     total += 1;
        // }
        if !inverted_list.is_empty() && inverted_list.len() <= min_cluster_size {
            //how_many += 1;
            to_be_replaced.extend(inverted_list.iter());
            inverted_list.clear();
        }
    }

    for &doc_id in to_be_replaced.iter() {
        let mut dense_vector: Vec<T> = vec![T::zero(); dataset.dim()];

        // Densify the vector
        for (&i, &v) in dataset.iter_vector(doc_id) {
            dense_vector[i as usize] = v;
        }

        let mut argmax = 0;
        let mut max = 0_f32;
        for (i, il) in inverted_lists.iter().enumerate() {
            if il.len() <= min_cluster_size {
                continue;
            }
            let (v_components, v_values) = dataset.get(i);

            let dot = dot_product_dense_sparse(&dense_vector, v_components, v_values);

            if dot > max {
                max = dot;
                argmax = i;
            }
        }

        inverted_lists[argmax].push(doc_id);
    }

    // println!("\tto_be_replaced: {}", to_be_replaced.len());
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
