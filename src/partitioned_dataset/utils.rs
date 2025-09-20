use std::hint::assert_unchecked;

use crate::ValueType;
use crate::{ComponentType, partitioned_dataset::fitting_integer::*};
use num_traits::One;
use num_traits::PrimInt;

pub fn map_components<const N_PARTITIONS: usize, const N_COMPONENT_BITS: usize>(
    partitions: &[FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>],
) -> Box<[FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>]>
where
    (): Fit<N_PARTITIONS>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>
        + Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS }>,
{
    // In theory, instead of usize, it should be `FittingInteger<N_COMPONENT_BITS.next_power_of_two().ilog2()>`,
    // but this is already messy enoguh...
    // Plus, the error checking at the end.
    let mut n_assigned_array = [0_usize; N_PARTITIONS];

    let mapping = partitions
        .iter()
        .map(|&partition| {
            let n_assigned = unsafe { n_assigned_array.get_unchecked_mut(partition.as_()) };
            let combined = partition.to_usize().unwrap() << N_COMPONENT_BITS | *n_assigned;
            *n_assigned += 1;
            FittingInteger::<
                    { N_PARTITIONS.next_power_of_two().ilog2() as usize + N_COMPONENT_BITS },
                >::from_usize(combined)
                .unwrap()
        })
        .collect();

    let actual_required_bit_shift = n_assigned_array
        .into_iter()
        .max()
        .unwrap()
        .next_power_of_two()
        .ilog2();
    assert_eq!(
        actual_required_bit_shift, N_COMPONENT_BITS as u32,
        "With {} partitions, this dataset requires N_COMPONENT_BITS = {}, but it was compiled with N_COMPONENT_BITS = {} instead.",
        N_PARTITIONS, actual_required_bit_shift, N_COMPONENT_BITS
    );

    mapping
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
