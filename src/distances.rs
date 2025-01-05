use std::cmp::Ordering;
use std::hint::assert_unchecked;

use crate::{ComponentType, ValueType};

// TODO: SAFETY: In the caller of these functions, assert safely that query_term_ids.len() == query_values.len()

/// Computes the dot product between a dense query and a sparse vector.
/// Before using this function, the query must be made dense. In some cases,
/// especially when queries have many non-zero components, this is faster
/// than computing the dot product with a "merge" style.
///
/// # Arguments
///
/// * `query` - The dense query vector.
/// * `v_term_ids` - The indices of the non-zero components in the vector.
/// * `v_values` - The values of the non-zero components in the vector.
///
/// # Returns
///
/// The dot product between the query and the vector.
///
/// # Examples
///
/// ```
/// use seismic::distances::dot_product_dense_sparse;
///
/// let query = [1.0, 2.0, 3.0, 0.0];
/// let v_term_ids = [0_u16, 2, 3];
/// let v_values = [1.0, 1.0, 1.5];
///
/// let result = dot_product_dense_sparse(&query, &v_term_ids, &v_values);
/// assert_eq!(result, 4.0);
/// ```
#[inline]
pub fn dot_product_dense_sparse<C, Q, V>(query: &[Q], v_term_ids: &[C], v_values: &[V]) -> f32
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    v_term_ids
        .iter()
        .zip(v_values)
        .map(|(&c, &v)| {
            // SAFETY: query.len() == dim + 1. This assumes the input is sanitized.
            unsafe { *query.get_unchecked(c.as_()) }
                .to_f32()
                .unwrap()
                .algebraic_mul(v.to_f32().unwrap())
        })
        .fold(0.0, |acc, x| acc.algebraic_add(x))
}

/// Computes the dot product between a query and a vector using merge style.
/// This function should be used when the query has just a few components.
/// Both the query's and vector's terms must be sorted by id.
///
/// # Arguments
///
/// * `query_term_ids` - The ids of the query terms.
/// * `query_values` - The values of the query terms.
/// * `v_term_ids` - The ids of the vector terms.
/// * `v_values` - The values of the vector terms.
///
/// # Returns
///
/// The dot product between the query and the vector.
///
/// # Examples
///
/// ```
/// use seismic::distances::dot_product_with_merge;
///
/// let query_term_ids = [1_u32, 2, 7];
/// let query_values = [1.0, 1.0, 1.0];
/// let v_term_ids = [0_u32, 1, 2, 3, 4];
/// let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];
///
/// let result = dot_product_with_merge(&query_term_ids, &query_values, &v_term_ids, &v_values);
/// assert_eq!(result, 2.0);
/// ```
#[inline]
#[must_use]
pub fn dot_product_with_merge<C, Q, V>(
    query_term_ids: &[C],
    query_values: &[Q],
    v_term_ids: &[C],
    v_values: &[V],
) -> f32
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    unsafe {
        assert_unchecked(
            v_term_ids.len() == v_values.len() && query_term_ids.len() == query_values.len(),
        )
    }
    let mut result = 0.0;
    let mut v_iter = v_term_ids.iter().zip(v_values);
    let mut current = v_iter.next();
    let b = current.is_some();
    for (&q_id, &q_v) in query_term_ids.iter().zip(query_values) {
        // This assert actually improves performance: https://github.com/rust-lang/rust/issues/134667
        if b {
            unsafe { assert_unchecked(current.is_some()) }
        }
        while let Some((&v_id, _)) = current
            && v_id < q_id
        {
            current = v_iter.next();
        }
        if !b {
            unsafe { assert_unchecked(current.is_none()) }
        }
        match current {
            Some((&v_id, v_v)) if v_id == q_id => {
                result += v_v.to_f32().unwrap() * q_v.to_f32().unwrap();
            }
            None => {
                break;
            }
            _ => {}
        }
    }
    result
}

/// Computes the dot product between a query and a vector using merge style.
/// (Alternative approach that uses less branches.)
/// This function should be used when the query has just a few components.
/// Both the query's and vector's terms must be sorted by id.
///
/// # Arguments
///
/// * `query_term_ids` - The ids of the query terms.
/// * `query_values` - The values of the query terms.
/// * `v_term_ids` - The ids of the vector terms.
/// * `v_values` - The values of the vector terms.
///
/// # Returns
///
/// The dot product between the query and the vector.
///
/// # Examples
///
/// ```
/// use seismic::distances::dot_product_with_merge_alt;
///
/// let query_term_ids = [1, 2, 7];
/// let query_values = [1.0, 1.0, 1.0];
/// let v_term_ids = [0, 1, 2, 3, 4];
/// let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];
///
/// let result = dot_product_with_merge_alt(&query_term_ids, &query_values, &v_term_ids, &v_values);
/// assert_eq!(result, 2.0);
/// ```
#[inline]
pub fn dot_product_with_merge_alt<Q, V>(
    query_term_ids: &[u16],
    query_values: &[Q],
    v_term_ids: &[u16],
    v_values: &[V],
) -> f32
where
    Q: ValueType,
    V: ValueType,
{
    unsafe {
        assert_unchecked(
            v_term_ids.len() == v_values.len() && query_term_ids.len() == query_values.len(),
        )
    }
    let mut result = 0.0;

    let (mut i, mut j) = (0, 0);

    loop {
        let (qt_i, vt_j) = unsafe {
            (
                *query_term_ids.get_unchecked(i),
                *v_term_ids.get_unchecked(j),
            )
        };
        i += if qt_i <= vt_j { 1 } else { 0 };
        j += if qt_i >= vt_j { 1 } else { 0 };
        if !(i < query_term_ids.len() && j < v_term_ids.len()) {
            break;
        }
        if query_term_ids[i] == v_term_ids[j] {
            result += query_values[i].to_f32().unwrap() * v_values[j].to_f32().unwrap();
        }
    }

    result
}

/// Computes the dot product between a sparse query and a sparse vector using binary search.
/// This function should be used when the query has just a few components.
/// Both the query's and vector's terms must be sorted by id.
///
/// # Arguments
///
/// * `query_term_ids` - The ids of the query terms.
/// * `query_values` - The values of the query terms.
/// * `v_term_ids` - The ids of the vector terms.
/// * `v_values` - The values of the vector terms.
///
/// # Returns
///
/// The dot product between the query and the vector.
///
/// # Examples
///
/// ```
/// use seismic::distances::dot_product_with_binary_search;
///
/// let query_term_ids = [1, 2, 7];
/// let query_values = [1.0, 1.0, 1.0];
/// let v_term_ids = [0, 1, 2, 3, 4];
/// let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];
///
/// let result = dot_product_with_binary_search(&query_term_ids, &query_values, &v_term_ids, &v_values);
/// assert_eq!(result, 2.0);
/// ```
#[inline]
pub fn dot_product_with_binary_search<Q, V>(
    query_term_ids: &[u16],
    query_values: &[Q],
    v_term_ids: &[u16],
    v_values: &[V],
) -> f32
where
    Q: ValueType,
    V: ValueType,
{
    unsafe {
        assert_unchecked(v_term_ids.len() == v_values.len());
    }
    query_term_ids
        .iter()
        .zip(query_values)
        .filter_map(|(t, v)| {
            binary_search(v_term_ids, t)
                .map(|i| v.to_f32().unwrap() * v_values[i].to_f32().unwrap())
        })
        .sum()
}

// Modified copy of std's implementation, that stops earlier and does linear searching for the rest.
// It's better by around 10-20%.
// Note: it seems that, for reason, using this can drastically hamper the performance of dot_product_with_merge... why?
#[inline]
fn binary_search<T>(arr: &[T], x: &T) -> Option<usize>
where
    T: Ord,
{
    binary_search_by(arr, |p| p.cmp(x))
}

#[inline]
fn binary_search_by<T, F>(arr: &[T], mut f: F) -> Option<usize>
where
    F: FnMut(&T) -> Ordering,
{
    const SIZE_STOP: usize = 7;

    let mut size = arr.len();
    let mut base = 0usize;

    while size > SIZE_STOP {
        let half = size / 2;
        let mid = base + half;

        let cmp = f(unsafe { arr.get_unchecked(mid) });

        // Sucks to be using intrinsics, but benchmarks show that this helps performance.
        base = std::intrinsics::select_unpredictable(cmp == Ordering::Greater, base, mid);

        size -= half;
    }

    unsafe { arr.get_unchecked(base..base + size) }
        .iter()
        .position(|x| f(x) == Ordering::Equal)
        .map(|pos| pos + base)
}
