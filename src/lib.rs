#![feature(iter_array_chunks)]
#![cfg_attr(
    all(feature = "prefetch", target_arch = "aarch64"),
    feature(stdarch_aarch64_prefetch)
)]
#![doc = include_str!("../README.md")]

use distances::{dot_product_dense_sparse, dot_product_dense_sparse_u32, dot_product_with_merge};
use pyo3::types::PyModuleMethods;

use fixed::types::extra::U8;
use fixed::FixedU16;
use fixed::FixedU8;
use half::bf16;
use half::f16;

pub mod pylib;

pub mod sparse_dataset;

use pylib::SeismicIndexRaw;
use pylib::SeismicIndexRawLV;
use pyo3::wrap_pyfunction;
pub use sparse_dataset::SparseDataset;
pub use sparse_dataset::SparseDatasetMut;

pub mod inverted_index;

pub use inverted_index::InvertedIndex;

pub mod inverted_index_wrapper;

pub use inverted_index_wrapper::SeismicDataset;
pub use inverted_index_wrapper::SeismicIndex;

pub mod quantized_summary;

pub use quantized_summary::QuantizedSummary;

pub mod elias_fano;

pub mod space_usage;

pub use space_usage::SpaceUsage;

pub mod distances;
pub mod json_utils;
pub mod topk_selectors;
pub mod utils;

use crate::pylib::get_seismic_string;
use crate::pylib::SeismicDataset as PySeismicDataset;
use crate::pylib::SeismicDatasetLV as PySeismicDatasetLV;

use crate::pylib::SeismicIndex as PySeismicIndex;
use crate::pylib::SeismicIndexLV as PySeismicIndexLV;

use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive, Zero};
use pyo3::prelude::PyModule;
use pyo3::{pymodule, Bound, PyResult};

/// Marker for types used as values in a dataset
pub trait ValueType:
    SpaceUsage
    + Copy
    + ToF32
    + FromPrimitive
    + ToPrimitive
    + Zero
    + Send
    + Sync
    + PartialOrd
    + FromPrimitive
{
}

impl ValueType for f64 {}

impl ValueType for f32 {}

impl ValueType for f16 {}

impl ValueType for bf16 {}

// Trait locale per conversione a f32
pub trait ToF32 {
    fn to_f32(self) -> f32;
}

impl ToF32 for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}
impl ToF32 for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl ToF32 for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}
impl ToF32 for bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}
impl ToF32 for FixedU8Q {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_num::<f32>()
    }
}
impl ToF32 for FixedU16Q {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_num::<f32>()
    }
}

/// Type aliases for quantized fixed-point types (8 fractional bits)
pub type FixedU8Q = FixedU8<U8>;
pub type FixedU16Q = FixedU16<U8>;

impl ValueType for FixedU8Q {}
impl ValueType for FixedU16Q {}

impl SpaceUsage for FixedU8Q {
    fn space_usage_byte(&self) -> usize {
        1 // 1 byte for FixedU8Q
    }
}

impl SpaceUsage for FixedU16Q {
    fn space_usage_byte(&self) -> usize {
        2 // 2 bytes for FixedU16Q
    }
}

pub trait ComponentType:
    AsPrimitive<usize>
    + FromPrimitive
    + SpaceUsage
    + Copy
    + Send
    + Sync
    + std::hash::Hash
    + Eq
    + Ord
    + std::convert::TryFrom<usize>
{
    /// Computes the dot product between a sparse query and a sparse vector.
    /// We need to explicitly specify the type of the query and the vector components
    /// for efficiency reasons. Relying on a generic ComponentType C entails
    /// the usage of the `as_()` method to cast components into `usize` in
    /// sparse-dense variant of the dot procuct computation. This has shown
    /// a performance degradation up to 10%.
    fn compute_dot_product<Q, V>(
        query: Option<&[Q]>,
        query_terms_ids: &[Self],
        query_values: &[Q],
        v_components: &[Self],
        v_values: &[V],
    ) -> f32
    where
        Q: ValueType,
        V: ValueType;
    const WIDTH: usize;
}

impl ComponentType for u16 {
    const WIDTH: usize = 16; // Required for compile time checks.

    /// Computes the dot product between a sparse query and a sparse vector.
    /// There are two cases:
    /// - `dense_query` is `Some`, which means the query was densified. This usually when the query has many non-zero components but the sparse space dimension is not too large. In this case, the `dot_product_dense_sparse` function is used.
    /// - `dense_query` is `None`, then a standard merge-based function is used to compute the dot product.
    ///
    /// # Arguments
    ///
    /// * `query` - The dense query vector.
    /// * `query_terms_ids` - The indices of the non-zero components in the query.
    /// * `query_values` - The values of the non-zero components in the query.
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
    /// let query_terms_ids = [0_u16, 1, 2];
    /// let query_values = [1.0, 2.0, 3.0];
    /// let v_components = [0_u16, 2, 3];
    /// let v_values = [1.0, 1.0, 1.5];
    ///
    /// let result = dot_product_dense_sparse(&query, &v_components, &v_values);
    /// assert_eq!(result, 4.0);
    /// ```
    #[inline]
    fn compute_dot_product<Q, V>(
        dense_query: Option<&[Q]>,
        query_terms_ids: &[u16],
        query_values: &[Q],
        v_components: &[u16],
        v_values: &[V],
    ) -> f32
    where
        Q: ValueType,
        V: ValueType,
    {
        if let Some(query) = dense_query {
            // This is the case when we have a dense query
            dot_product_dense_sparse(query, v_components, v_values)
        } else {
            dot_product_with_merge(query_terms_ids, query_values, v_components, v_values)
        }
    }
}

//impl ComponentType for u16 {}

impl ComponentType for u32 {
    const WIDTH: usize = 32; // Required for compile time checks.

    #[inline]
    fn compute_dot_product<Q, V>(
        dense_query: Option<&[Q]>,
        query_terms_ids: &[u32],
        query_values: &[Q],
        v_components: &[u32],
        v_values: &[V],
    ) -> f32
    where
        Q: ValueType,
        V: ValueType,
    {
        if let Some(query) = dense_query {
            // This is the case when we have a dense query
            dot_product_dense_sparse_u32(query, v_components, v_values)
        } else {
            dot_product_with_merge(query_terms_ids, query_values, v_components, v_values)
        }
    }
}

/// A Python module implemented in Rust. The name of this function must match the `lib.name`
/// setting in the `Cargo.toml`, otherwise Python will not be able to import the module.
#[pymodule]
fn seismic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_seismic_string, m)?)?;
    m.add_class::<PySeismicIndex>()?;
    m.add_class::<PySeismicIndexLV>()?;
    m.add_class::<SeismicIndexRaw>()?;
    m.add_class::<SeismicIndexRawLV>()?;
    m.add_class::<PySeismicDataset>()?;
    m.add_class::<PySeismicDatasetLV>()?;
    Ok(())
}
