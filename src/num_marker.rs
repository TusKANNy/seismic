// This whole file exists as a workaround to Rust's deficiencies regarding trait implementation conflicts.
// It's very ugly.

use fixed::{
    FixedU8, FixedU16, FixedU32, FixedU64, FixedU128,
    traits::{FixedUnsigned, ToFixed},
};
use half::{bf16, f16};
use num_traits::AsPrimitive;

use crate::ValueType;

pub(crate) trait MarkerFixedSigned: FixedUnsigned + ValueType {}
impl<T: fixed::types::extra::LeEqU8 + Send + Sync> MarkerFixedSigned for FixedU8<T> {}
impl<T: fixed::types::extra::LeEqU16 + Send + Sync> MarkerFixedSigned for FixedU16<T> {}
impl<T: fixed::types::extra::LeEqU32 + Send + Sync> MarkerFixedSigned for FixedU32<T> {}
impl<T: fixed::types::extra::LeEqU64 + Send + Sync> MarkerFixedSigned for FixedU64<T> {}
impl<T: fixed::types::extra::LeEqU128 + Send + Sync> MarkerFixedSigned for FixedU128<T> {}

pub trait FromF32 {
    fn from_f32_saturating(n: f32) -> Self;
}

impl<T> FromF32 for T
where
    T: MarkerFixedSigned,
{
    fn from_f32_saturating(n: f32) -> Self {
        n.saturating_to_fixed()
    }
}

macro_rules! impl_from_f32_saturating {
    ($($t:ty),*) => {
        $(impl FromF32 for $t {
            fn from_f32_saturating(n: f32) -> Self {
                n.as_()
            }
        })*
    }
}

impl_from_f32_saturating![f64, f32, f16, bf16];
