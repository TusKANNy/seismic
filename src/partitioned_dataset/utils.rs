use std::hash::{DefaultHasher, Hash, Hasher};
use std::hint::assert_unchecked;
use std::time::Instant;

use crate::sparse_dataset::SparseDatasetGeneric;
use crate::utils::{read_from_path, write_to_path};
use crate::{ComponentType, partitioned_dataset::fitting_integer::*};
use crate::{SpaceUsage, ValueType};
use metis::Graph;
use num_traits::PrimInt;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

/// A symmetrical matrix where the diagonal components are 0. Only the upper part of the matrix is stored.
pub struct HollowSymmetricMatrix<T: Zero + Copy> {
    dim: usize,
    data: Box<[T]>,
}

impl<T: Zero + Copy> HollowSymmetricMatrix<T> {
    pub fn new(dim: usize) -> Self {
        let size = (dim * (dim - 1)) / 2;
        let data = vec![T::zero(); size].into_boxed_slice();
        Self { dim, data }
    }

    /// # Safety
    /// - `j > i`
    /// - `i < self.dim && j < self.dim`
    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        unsafe {
            assert_unchecked(j > i);
            let index = i * self.dim + j - ((i + 2) * (i + 1)) / 2;
            self.data.get_unchecked(index)
        }
    }

    /// # Safety
    /// - `j > i`
    /// - `i < self.dim && j < self.dim`
    pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        unsafe {
            assert_unchecked(j > i);
            let index = i * self.dim + j - ((i + 2) * (i + 1)) / 2;
            self.data.get_unchecked_mut(index)
        }
    }

    /// Iterates the specified row (how it's supposed to be, not how it's represented).
    /// The diagonal element is skipped.
    pub fn iter_row(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let before = (0..i).map(move |j| (j, unsafe { self.get_unchecked(j, i) }));
        let after = ((i + 1)..self.dim).map(move |j| (j, unsafe { self.get_unchecked(i, j) }));
        before.chain(after)
    }
}

#[derive(Serialize, Deserialize)]
pub struct MetisParams {
    pub adjncy: Box<[i32]>,
    pub weights: Box<[i32]>,
    pub xadj: Box<[i32]>,
}

impl MetisParams {
    pub fn build_partitions<const N_PARTITIONS: usize>(
        &self,
    ) -> Vec<FittingInteger<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>>
    where
        (): Fit<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>,
    {
        print!("\tBuilding partitions ");
        let time = Instant::now();

        let mut part = vec![0; self.xadj.len() - 1];

        Graph::new(1, N_PARTITIONS as i32, &self.xadj, &self.adjncy)
            .unwrap()
            .set_adjwgt(&self.weights)
            .part_recursive(part.as_mut_slice())
            .unwrap();

        let result = part
            .into_iter()
            .map(|p| {
                FittingInteger::<{ N_PARTITIONS.next_power_of_two().ilog2() as usize }>::from_i32(p)
                    .unwrap()
            })
            .collect();

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        result
    }
}

/// Load the dataset's adjacency matrix
///
/// Hash the dataset's components and offsets so that its adjacency can be cached (as it's a very long operation)
pub fn build_or_load_metis_params<C, V, O, AC, AV>(
    dataset: &SparseDatasetGeneric<C, V, O, AC, AV>,
) -> MetisParams
where
    C: ComponentType,
    V: ValueType,
    O: AsRef<[usize]> + SpaceUsage + Hash,
    AC: AsRef<[C]> + SpaceUsage + Hash,
    AV: AsRef<[V]> + SpaceUsage,
{
    let mut s = DefaultHasher::new();
    dataset.components().hash(&mut s);
    dataset.offsets().hash(&mut s);
    let hash = s.finish();
    let filename = format!("cached_adjacency_{}", hash);
    if !std::fs::exists(filename.as_str()).is_ok_and(|b| b) {
        println!("Adjacency matrix not cached. Creating.");
        let params = dataset.adjacency_matrix_metis();

        println!("Saving ... {}", filename);
        write_to_path(&params, filename.as_str()).unwrap();

        params
    } else {
        println!("Loading adjacency matrix {}.", filename.as_str());
        read_from_path(filename.as_str()).unwrap()
    }
}

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
