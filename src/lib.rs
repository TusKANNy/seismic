#![feature(iter_array_chunks)]
#![cfg_attr(
    all(feature = "prefetch", target_arch = "aarch64"),
    feature(stdarch_aarch64_prefetch)
)]
#![doc = include_str!("../README.md")]
use pyo3::types::PyModuleMethods;

use half::f16;

pub mod pylib;

pub mod sparse_dataset;

use pylib::SeismicIndexRaw;
pub use sparse_dataset::SparseDataset;
pub use sparse_dataset::SparseDatasetMut;

pub mod inverted_index;

pub use inverted_index::InvertedIndex;

pub mod inverted_index_wrapper;

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

use crate::pylib::SeismicIndex as PySeismicIndex;
use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive, Zero};
use pyo3::prelude::PyModule;
use pyo3::{pymodule, Bound, PyResult};

/// Marker for types used as values in a dataset
pub trait DataType:
    SpaceUsage + Copy + AsPrimitive<f16> + ToPrimitive + Zero + Send + Sync + PartialOrd + FromPrimitive
{
}

impl DataType for f64 {}

impl DataType for f32 {}

impl DataType for f16 {}

/// A Python module implemented in Rust. The name of this function must match the `lib.name`
/// setting in the `Cargo.toml`, otherwise Python will not be able to import the module.
#[pymodule]
fn seismic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeismicIndex>()?;
    m.add_class::<SeismicIndexRaw>()?;
    Ok(())
}
