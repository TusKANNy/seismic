//! Minimal implementation of Elias-Fano encoding.
//! This implementation is inspired by C++ implementation by [Giuseppe Ottaviano](https://github.com/ot/succinct/blob/master/elias_fano.hpp).
//! (https://github.com/kampersanda/sucds/blob/main/src/mii_sequences/elias_fano.rs)

use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use qwt::bitvector::{BitVector, BitVectorMut};
use qwt::darray::DArray;
use qwt::utils::msb;
use qwt::SelectBin;

/// Compressed monotone increasing sequence through Elias-Fano encoding.
///
/// # Example
/// ```
/// use seismic::elias_fano::EliasFano;
///
/// let v: Vec<usize> = vec![1,3,3,7];
/// let ef = EliasFano::from(&v);
///
/// assert_eq!(ef.len(), 4);
/// assert_eq!(ef.universe(), 8);
///
/// assert_eq!(ef.select(0), Some(1));
/// assert_eq!(ef.select(1), Some(3));
/// assert_eq!(ef.select(2), Some(3));
/// assert_eq!(ef.select(3), Some(7));
/// ```
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EliasFano {
    high_bits: DArray<false>,
    low_bits: BitVector,
    low_len: usize,
    universe: usize,
    num_vals: usize,
}

impl EliasFano {
    /// Builds a Elias-Fano sequence from a slice of values.
    ///
    /// # Panic
    /// Panics if the sequence is not monotonically increasing.
    ///
    /// # Example
    /// ```
    /// use seismic::elias_fano::EliasFano;
    ///
    /// let v1: Vec<usize> = vec![1,3,3,7];
    /// let ef1 = EliasFano::from(&v1);
    /// assert_eq!(ef1.len(), 4);
    /// ```
    pub fn from(data: &[usize]) -> EliasFano {
        if data.is_empty() {
            return EliasFano::default();
        }

        assert!(
            data.windows(2).all(|w| w[0] <= w[1]),
            "The sequence must be monotonically increasing."
        );

        let mut efb = EliasFanoBuilder::new(1 + data.last().unwrap(), data.len());

        efb.extend(data.to_vec());

        efb.build()
    }

    /// Returns the position of the `k`-th smallest integer, or
    /// [`None`] if `k > self.num_vals`.
    ///
    /// # Example
    /// ```
    /// use seismic::elias_fano::EliasFano;
    ///
    /// let v: Vec<usize> = vec![1,3,3,7];
    /// let ef = EliasFano::from(&v);
    ///
    /// assert_eq!(ef.select(0), Some(1));
    /// assert_eq!(ef.select(1), Some(3));
    /// assert_eq!(ef.select(2), Some(3));
    /// assert_eq!(ef.select(3), Some(7));
    /// assert_eq!(ef.select(4), None);
    /// ```
    #[inline(always)]
    pub fn select(&self, k: usize) -> Option<usize> {
        if k >= self.num_vals {
            return None;
        }
        Some(
            ((self.high_bits.select1(k)? - k) << self.low_len)
                | if self.low_len > 0 {
                    unsafe {
                        self.low_bits
                            .get_bits_unchecked(k * self.low_len, self.low_len)
                            as usize
                    }
                } else {
                    0
                },
        )
    }

    /// Checks if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vals == 0
    }

    /// Returns the length of the compressed sequence.
    ///
    /// ## Examples
    /// ```
    /// use seismic::elias_fano::EliasFano;
    ///
    /// let v: Vec<usize> = vec![1,3,3,7];
    /// let ef = EliasFano::from(&v);
    ///
    /// assert_eq!(ef.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.num_vals
    }

    /// Returns the universe, i.e., the (exclusive) upper bound of possible integers.
    ///
    /// ## Examples
    /// ```
    /// use seismic::elias_fano::EliasFano;
    ///
    /// let v: Vec<usize> = vec![1,3,3,7];
    /// let ef = EliasFano::from(&v);
    ///
    /// assert_eq!(ef.universe(), 8);
    /// ```
    pub fn universe(&self) -> usize {
        self.universe
    }
}

/// Builder for [`EliasFano`].
///
/// # Examples
/// ```
/// use seismic::elias_fano::{EliasFano, EliasFanoBuilder};
///
/// let v: Vec<usize> = vec![1,3,3,7];
/// let mut efb = EliasFanoBuilder::new(8,4);
///
/// for &n in v.iter(){
///     let _ = efb.push(n);
/// }
///
/// let ef = efb.build();
/// let ef = EliasFano::from(&v);
///
/// assert_eq!(ef.len(), 4);
/// assert_eq!(ef.universe(), 8);
/// ```
pub struct EliasFanoBuilder {
    high_bits: BitVectorMut,
    low_bits: BitVectorMut,
    universe: usize,
    num_vals: usize,
    pos: usize,
    last: usize,
    low_len: usize,
}

impl EliasFanoBuilder {
    /// Creates a new, empty, Elias-Fano builder to store n values up to u (u excluded).
    ///
    /// # Panics
    /// Panics if `num_vals` is zero.
    pub fn new(universe: usize, num_vals: usize) -> Self {
        assert!(
            num_vals > 0,
            "The number of values num_vals must not be zero."
        );

        let low_len = msb(universe / num_vals) as usize;

        Self {
            high_bits: BitVectorMut::with_zeros((num_vals + 1) + (universe >> low_len) + 1),
            low_bits: BitVectorMut::new(),
            universe,
            num_vals,
            pos: 0,
            last: 0,
            low_len,
        }
    }

    /// Pushes integer `val` at the end.
    ///
    /// # Panics
    /// Panics if
    /// - `val` is less than the last one,
    /// - `val` is no less than [`Self::universe()`], or
    /// - the number of stored integers becomes no less than [`Self::num_vals()`].
    pub fn push(&mut self, val: usize) {
        assert!(
            self.last <= val,
            "val must be no less than the last inserted one {}, but got {val}.",
            self.last
        );

        assert!(
            val < self.universe,
            "val must be less than self.universe()={}, but got {val}.",
            self.universe
        );
        assert!(
            self.pos < self.num_vals,
            "The number of pushed integers must not exceed self.num_vals()={}.",
            self.num_vals
        );

        self.last = val;
        let low_mask = (1 << self.low_len) - 1;
        if self.low_len != 0 {
            self.low_bits
                .append_bits(val as u64 & low_mask, self.low_len);
        }
        self.high_bits.set((val >> self.low_len) + self.pos, true);
        self.pos += 1;
    }

    /// Appends integers at the end.
    ///
    /// # Panics
    /// Panics if
    /// - `vals` is not monotone increasing (also compared to the current last value),
    /// - values in `vals` is no less than [`Self::universe()`], or
    /// - the number of stored integers becomes no less than [`Self::num_vals()`].
    pub fn extend<I>(&mut self, vals: I)
    where
        I: IntoIterator<Item = usize>,
    {
        for x in vals {
            self.push(x);
        }
    }

    /// Builds [`EliasFano`] from the pushed integers.
    pub fn build(self) -> EliasFano {
        EliasFano {
            high_bits: DArray::new(self.high_bits.into()),
            low_bits: self.low_bits.into(),
            num_vals: self.num_vals,
            low_len: self.low_len,
            universe: self.universe,
        }
    }

    /// Returns the universe, i.e., the (exclusive) upper bound of possible integers.
    #[inline(always)]
    pub const fn universe(&self) -> usize {
        self.universe
    }

    /// Returns the number of integers that can be stored.
    #[inline(always)]
    pub const fn num_vals(&self) -> usize {
        self.num_vals
    }
}

use qwt::SpaceUsage as QwtSpaceUsage;

impl SpaceUsage for EliasFano {
    /// Returns the space usage in bytes.
    fn space_usage_byte(&self) -> usize {
        self.high_bits.space_usage_byte()
            + self.low_bits.space_usage_byte()
            + QwtSpaceUsage::space_usage_byte(&self.low_len)
            + SpaceUsage::space_usage_byte(&self.universe)
            + SpaceUsage::space_usage_byte(&self.num_vals)
    }
}

#[cfg(test)]
mod tests;
