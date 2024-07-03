use crate::{utils::binary_search_branchless, DataType};

/// Computes the dot product between a dense query and a sparse vector.
/// Before using this function, the query must be made dense. This is much faster
/// than computing the dot product with a "merge" style.
///
/// # Arguments
///
/// * `query` - The dense query vector.
/// * `v_components` - The indices of the non-zero components in the vector.
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
/// let v_components = [0, 2, 3];
/// let v_values = [1.0, 1.0, 1.5];
///
/// let result = dot_product_dense_sparse(&query, &v_components, &v_values);
/// assert_eq!(result, 4.0);
/// ```
#[inline]
#[must_use]
pub fn dot_product_dense_sparse<Q, V>(query: &[Q], v_components: &[u16], v_values: &[V]) -> f32
where
    Q: DataType,
    V: DataType,
{
    const N_LANES: usize = 4;

    let mut result = [0.0; N_LANES];
    let chunk_iter = v_components.iter().zip(v_values).array_chunks::<N_LANES>();

    for chunk in chunk_iter {
        //for i in 0..N_LANES { // Slightly faster withour this for.
        result[0] += query[*chunk[0].0 as usize].to_f32().unwrap() * (chunk[0].1.to_f32().unwrap());
        result[1] += query[*chunk[1].0 as usize].to_f32().unwrap() * chunk[1].1.to_f32().unwrap();
        result[2] += query[*chunk[2].0 as usize].to_f32().unwrap() * chunk[2].1.to_f32().unwrap();
        result[3] += query[*chunk[3].0 as usize].to_f32().unwrap() * chunk[3].1.to_f32().unwrap();
        //result[3] += unsafe { *query.get_unchecked(*chunk[3].0 as usize) } * *chunk[3].1;
        //}
    }

    let l = v_components.len();
    let rem = l % N_LANES;

    if rem > 0 {
        for (&i, &v) in v_components[l - rem..].iter().zip(&v_values[l - rem..]) {
            result[0] += query[i as usize].to_f32().unwrap() * v.to_f32().unwrap();
        }
    }

    result.iter().sum()

    // This is what we would like to write :-)
    // let mut result = 0.0;
    // for (&i, &v) in self.iter_vector(id) {
    //     result += query[i as usize] * v;
    // }

    // result
}

/// Computes the dot product between a sparse query and a sparse vector using binary search.
/// This function should be used when the query has just a few components.
/// Both the query's and vector's terms must be sorted by id.
///
/// # Arguments
///
/// * `query_term_ids` - The ids of the query terms.
/// * `query_values` - The values of the query terms.
/// * `v_terms_ids` - The ids of the vector terms.
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
#[must_use]
pub fn dot_product_with_binary_search<Q, V>(
    query_term_ids: &[u16],
    query_values: &[Q],
    v_terms_ids: &[u16],
    v_values: &[V],
) -> f32
where
    Q: DataType,
    V: DataType,
{
    let mut result = 0.0;

    for (&term_id, &value) in query_term_ids.iter().zip(query_values) {
        // Let's use a branchless binary search
        let i = binary_search_branchless(v_terms_ids, term_id);

        // SAFETY: result of binary search is always smaller than v_term_id.len() and v_values.len()
        let cmp = *unsafe { v_terms_ids.get_unchecked(i) } == term_id;
        result += if cmp {
            value.to_f32().unwrap() * unsafe { v_values.get_unchecked(i).to_f32().unwrap() }
        } else {
            0.0
        };
    }
    result
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
/// let query_term_ids = [1, 2, 7];
/// let query_values = [1.0, 1.0, 1.0];
/// let v_term_ids = [0, 1, 2, 3, 4];
/// let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];
///
/// let result = dot_product_with_merge(&query_term_ids, &query_values, &v_term_ids, &v_values);
/// assert_eq!(result, 2.0);
/// ```
#[inline]
#[must_use]
pub fn dot_product_with_merge<Q, V>(
    query_term_ids: &[u16],
    query_values: &[Q],
    v_term_ids: &[u16],
    v_values: &[V],
) -> f32
where
    Q: DataType,
    V: DataType,
{
    let mut result = 0.0;
    let mut i = 0;
    for (&q_id, &q_v) in query_term_ids.iter().zip(query_values) {
        unsafe {
            while i < v_term_ids.len() && *v_term_ids.get_unchecked(i) < q_id {
                i += 1;
            }

            if i == v_term_ids.len() {
                break;
            }

            if *v_term_ids.get_unchecked(i) == q_id {
                result += (*v_values.get_unchecked(i)).to_f32().unwrap() * q_v.to_f32().unwrap();
            }
        }
    }
    result
}
