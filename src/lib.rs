#![doc = include_str!("../README.md")]

pub use vectorium::{SparseDataset, SparseDatasetGrowable};

pub mod configurations;
pub(crate) mod posting_list;

pub mod inverted_index;
pub use inverted_index::InvertedIndex;

pub mod inverted_index_wrapper;
pub use inverted_index_wrapper::SeismicDataset;
pub use inverted_index_wrapper::SeismicIndex;

pub mod quantized_summary;
pub use quantized_summary::QuantizedSummary;

pub use vectorium::{ComponentType, FromF32, SpaceUsage, ValueType};

pub mod json_utils;
pub mod utils;

/// Type aliases for quantized fixed-point types (re-exported from vectorium).
pub use vectorium::{FixedU8Q, FixedU16Q};

#[cfg(feature = "python")]
pub mod pylib;

#[cfg(feature = "python")]
use {
    pylib::SeismicDataset as PySeismicDataset,
    pylib::SeismicDatasetLV as PySeismicDatasetLV,
    pylib::SeismicIndex as PySeismicIndex,
    pylib::SeismicIndexLV as PySeismicIndexLV,
    pylib::SeismicIndexRaw,
    pylib::SeismicIndexRawLV,
    pylib::get_seismic_string,
    pyo3::prelude::PyModule,
    pyo3::prelude::*,
    pyo3::{Bound, PyResult, pymodule},
};

#[cfg(feature = "python")]
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
