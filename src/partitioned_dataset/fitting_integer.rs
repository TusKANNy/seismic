use bytemuck::Pod;
use num_traits::{FromBytes, FromPrimitive, One, ToBytes, ToPrimitive};
use serde::{Serialize, de::DeserializeOwned};

use crate::SpaceUsage;

// This is a very ugly workaround to achieve the equivalent of the following:
// ```
// type FittingInteger<const N: usize> = CondType<{N <= 8}, u8, CondType<{N <= 16}, u16, CondType<{N <= 32}, u32, usize>>>
// ```
//
// However, as described by https://github.com/nvzqz/condtype?tab=readme-ov-file#limitations, this is not usable until
// https://github.com/rust-lang/project-const-generics/issues/26 is fixed.
//
// TODO: when it is, remove this ugly workaround, and all the ugly `where`'s in partitioned_dataset.rs related to it!
pub trait Fit<const N: usize> {
    type Integer: num_traits::PrimInt
        + SpaceUsage
        + FromPrimitive
        + ToPrimitive
        + FromBytes
        + ToBytes
        + One
        + Send
        + Sync
        + Pod
        + Serialize
        + DeserializeOwned;
}

pub type FittingInteger<const N: usize> = <() as Fit<N>>::Integer;

pub type FittingArray<const N: usize>
where
    (): Fit<N>,
= [FittingInteger<N>; N.div_ceil(size_of::<FittingInteger<N>>() * 8)];

macro_rules! impl_fit {
    ($($n:expr => $ty:ty),* $(,)?) => {
        $(
            impl Fit<$n> for () {
                type Integer = $ty;
            }
        )*
    };
}

impl_fit!(
    1 => u8,
    2 => u8,
    3 => u8,
    4 => u8,
    5 => u8,
    6 => u8,
    7 => u8,
    8 => u8,
    9 => u16,
    10 => u16,
    11 => u16,
    12 => u16,
    13 => u16,
    14 => u16,
    15 => u16,
    16 => u16,
    17 => u32,
    18 => u32,
    19 => u32,
    20 => u32,
    21 => u32,
    22 => u32,
    23 => u32,
    24 => u32,
    25 => u32,
    26 => u32,
    27 => u32,
    28 => u32,
    29 => u32,
    30 => u32,
    31 => u32,
    32 => u32,
    33 => usize,
    34 => usize,
    35 => usize,
    36 => usize,
    37 => usize,
    38 => usize,
    39 => usize,
    40 => usize,
    41 => usize,
    42 => usize,
    43 => usize,
    44 => usize,
    45 => usize,
    46 => usize,
    47 => usize,
    48 => usize,
    49 => usize,
    50 => usize,
    51 => usize,
    52 => usize,
    53 => usize,
    54 => usize,
    55 => usize,
    56 => usize,
    57 => usize,
    58 => usize,
    59 => usize,
    60 => usize,
    61 => usize,
    62 => usize,
    63 => usize,
    64 => usize,
    65 => usize,
    66 => usize,
    67 => usize,
    68 => usize,
    69 => usize,
    70 => usize,
    71 => usize,
    72 => usize,
    73 => usize,
    74 => usize,
    75 => usize,
    76 => usize,
    77 => usize,
    78 => usize,
    79 => usize,
    80 => usize,
    81 => usize,
    82 => usize,
    83 => usize,
    84 => usize,
    85 => usize,
    86 => usize,
    87 => usize,
    88 => usize,
    89 => usize,
    90 => usize,
    91 => usize,
    92 => usize,
    93 => usize,
    94 => usize,
    95 => usize,
    96 => usize,
    97 => usize,
    98 => usize,
    99 => usize,
    100 => usize,
    101 => usize,
    102 => usize,
    103 => usize,
    104 => usize,
    105 => usize,
    106 => usize,
    107 => usize,
    108 => usize,
    109 => usize,
    110 => usize,
    111 => usize,
    112 => usize,
    113 => usize,
    114 => usize,
    115 => usize,
    116 => usize,
    117 => usize,
    118 => usize,
    119 => usize,
    120 => usize,
    121 => usize,
    122 => usize,
    123 => usize,
    124 => usize,
    125 => usize,
    126 => usize,
    127 => usize,
    128 => usize,
);
