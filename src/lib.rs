#![allow(refining_impl_trait)]
#![feature(array_windows)]
#![feature(core_intrinsics)]
#![feature(float_algebraic)]
#![feature(gen_blocks)]
#![feature(generic_const_exprs)]
#![feature(iter_map_windows)]
#![feature(portable_simd)]
#![feature(slice_as_array)]
#![feature(trait_alias)]
#![feature(vec_into_chunks)]
#![feature(vec_push_within_capacity)]
#![doc = include_str!("../README.md")]

use fixed::FixedU8;
use fixed::FixedU16;
use num_traits::PrimInt;
use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive, Unsigned, Zero};

pub mod sparse_dataset;
use std::fmt::Debug;
use std::hash::Hash;

use bytemuck::Pod;
pub use sparse_dataset::FromDatasetGenericF32;
pub use sparse_dataset::SparseDataset;
pub use sparse_dataset::SparseDatasetMut;
pub use sparse_dataset::SparseDatasetTrait;

pub mod compressed_dataset;
pub use compressed_dataset::PermutationStrategy;
pub mod partitioned_dataset;

pub mod baseline_streamvbyte_dataset;
pub use crate::baseline_streamvbyte_dataset::dataset::BaselineStreamVByteDataset;

pub mod inverted_index;
pub use inverted_index::InvertedIndex;

pub mod inverted_index_wrapper;
pub use inverted_index_wrapper::SeismicDataset;
pub use inverted_index_wrapper::SeismicIndex;

pub mod quantized_summary;
pub use quantized_summary::QuantizedSummary;

pub mod stream_vbyte_dataset;

mod num_marker;

pub mod space_usage;
pub use space_usage::SpaceUsage;

pub mod distances;

pub mod json_utils;
pub mod utils;

pub mod libbin;

use crate::num_marker::FromF32;

/// Type aliases for quantized fixed-point types. You can change FRAC in the `fixed` crate to adjust the precision.
/// The `FixedU8Q` type uses 6 fractional bits, while `FixedU16Q` uses 8 fractional bits.
use fixed::types::extra::U6;
use fixed::types::extra::U13;
pub type FixedU8Q = FixedU8<U6>;
pub type FixedU16Q = FixedU16<U13>;

/// Marker for types used as values in a dataset
pub trait ValueType = SpaceUsage
    + Copy
    + ToPrimitive
    + Zero
    + Send
    + Sync
    + PartialOrd
    + FromPrimitive
    + FromF32
    + Pod
    + 'static;

pub trait ComponentType = Unsigned
    + PrimInt
    + AsPrimitive<usize>
    + FromPrimitive
    + ToPrimitive
    + Unsigned
    + SpaceUsage
    + Copy
    + Debug
    + Default
    + Send
    + Sync
    + Hash
    + Eq
    + Ord
    + Pod;

#[cfg(feature = "pyo3")]
pub mod pylib;

#[cfg(feature = "pyo3")]
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

#[cfg(feature = "pyo3")]
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
