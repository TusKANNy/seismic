use half::f16;
use std::mem;

/// A trait to report the space usage of a data structure.
pub trait SpaceUsage {
    /// Gives the space usage of the data structure in bytes.
    fn space_usage_byte(&self) -> usize;

    /// Gives the space usage of the data structure in KiB.
    #[allow(non_snake_case)]
    fn space_usage_KiB(&self) -> f64 {
        let bytes = self.space_usage_byte();
        (bytes as f64) / (1024_f64)
    }

    /// Gives the space usage of the data structure in MiB.
    #[allow(non_snake_case)]
    fn space_usage_MiB(&self) -> f64 {
        let bytes = self.space_usage_byte();
        (bytes as f64) / ((1024 * 1024) as f64)
    }

    /// Gives the space usage of the data structure in GiB.
    #[allow(non_snake_case)]
    fn space_usage_GiB(&self) -> f64 {
        let bytes = self.space_usage_byte();
        (bytes as f64) / ((1024 * 1024 * 1024) as f64)
    }
}

/// TODO: Improve and generalize. Incorrect if T is not a primitive type.
/// It is also error-prone to implement this for every data structure.
/// Make a macro to go over the member of a struct!
impl<T> SpaceUsage for Vec<T>
where
    T: Copy,
{
    fn space_usage_byte(&self) -> usize {
        mem::size_of::<Self>() + mem::size_of::<T>() * self.capacity()
    }
}

impl<T> SpaceUsage for Box<[T]>
where
    T: SpaceUsage + Copy,
{
    fn space_usage_byte(&self) -> usize {
        if !self.is_empty() {
            mem::size_of::<Self>() + self.first().unwrap().space_usage_byte() * self.len()
        } else {
            mem::size_of::<Self>()
        }
    }
}

macro_rules! impl_space_usage {
    ($($t:ty),*) => {
        $(impl SpaceUsage for $t {
            fn space_usage_byte(&self) -> usize {
                mem::size_of::<Self>()
            }
        })*
    }
}

impl_space_usage![
    bool, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize, f32, f64, f16
];
