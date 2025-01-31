use std::hint::assert_unchecked;

use crate::{ComponentType, partitioned_dataset::fitting_integer::*};
use itertools::Itertools;
use num_traits::PrimInt;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn partitioning_function<const N_PARTITIONS: usize, C: ComponentType>(n: C) -> usize {
    n.as_() % N_PARTITIONS
}

/// Given an array of `bool`s representing the adctive partitions, generates a bitset representing it.
pub(crate) fn gen_active_partitions<const N_PARTITIONS: usize>(
    active_partitions_iter: [bool; N_PARTITIONS],
) -> FittingArray<N_PARTITIONS>
where
    (): Fit<N_PARTITIONS>,
{
    unsafe { assert_unchecked(active_partitions_iter.len() == N_PARTITIONS) };
    let leftmost_one = FittingInteger::<N_PARTITIONS>::one().rotate_right(1);
    let mut iter = active_partitions_iter.into_iter();
    std::array::from_fn(|_| {
        let mut active_partitions = FittingInteger::<N_PARTITIONS>::zero();
        for (i, a) in iter
            .by_ref()
            .take(size_of::<FittingInteger<N_PARTITIONS>>() * 8)
            .enumerate()
        {
            if a {
                active_partitions = active_partitions | (leftmost_one >> i);
            }
        }
        active_partitions
    })
}

pub(crate) fn sort_by_partition<F: Fn(C) -> usize, C: Clone, V>(
    components_values: impl Iterator<Item = (C, V)>,
    partitioning_function: F,
) -> (Vec<C>, Vec<V>) {
    components_values
        .sorted_by_key(|(c, _)| partitioning_function(c.clone()))
        .unzip()
}

#[inline]
pub(crate) fn partitions_len_array<const N_PARTITIONS: usize, F: Fn(C) -> usize, C>(
    components: impl Iterator<Item = C>,
    partitioning_function: F,
) -> [usize; N_PARTITIONS] {
    let mut partitions_len = [0; N_PARTITIONS];

    for c in components {
        unsafe {
            *partitions_len.get_unchecked_mut(partitioning_function(c)) += 1;
        }
    }

    partitions_len
}

#[inline]
pub(crate) fn count_ones_array<T: PrimInt>(arr: &[T]) -> u32 {
    arr.iter().map(|n| n.count_ones()).sum()
}
