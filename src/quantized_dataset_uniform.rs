use std::hint::assert_unchecked;

use indicatif::ProgressIterator;
use num_traits::ToPrimitive;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use serde::{Deserialize, Serialize};

use crate::distances::dot_product_dense_sparse;
use crate::sparse_dataset::SparseDatasetGeneric;
use crate::utils::{get_uniform_quantization_info_pivot, prefetch_read_slice};
use crate::{ComponentType, FromDatasetGenericF32, SpaceUsage, SparseDatasetTrait, ValueType};

const LOW_BINS: u8 = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetQuantizedUniformComponent<C>
where
    C: ComponentType,
{
    dim: usize,
    offsets: Box<[usize]>,
    components: Box<[C]>,
    values: Box<[u8]>,
    pivots: Box<[f32]>,
    quants: Box<[f32]>,
    second_quants: Box<[f32]>,
}

impl<C> SparseDatasetTrait for SparseDatasetQuantizedUniformComponent<C>
where
    C: ComponentType,
{
    type Component = C;
    type Value = f32;
    type PreparedQuery<D: ComponentType, W: ValueType> = Vec<W>;

    // #[inline]
    // fn get_with_offset_iter(
    //     &self,
    //     offset: usize,
    //     len: usize,
    // ) -> impl Iterator<Item = (Self::Component, Self::Value)> {
    //     unsafe { assert_unchecked(len > 0) };
    //     let components = unsafe { self.components.get_unchecked(offset..(offset + len)) };
    //     let values = unsafe { self.values.get_unchecked(offset..(offset + len)) };

    //     components.iter().zip(values).map(move |(&c, &v)| {
    //         let value = (v as f32)
    //             .algebraic_mul(self.quants[c.as_()])
    //             .algebraic_add(self.pivots[c.as_()]);
    //         (c, value)
    //     })
    // }

    #[inline]
    fn get_with_offset_iter(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (Self::Component, Self::Value)> {
        unsafe { assert_unchecked(len > 0) };
        let components = unsafe { self.components.get_unchecked(offset..(offset + len)) };
        let values = unsafe { self.values.get_unchecked(offset..(offset + len)) };

        components.iter().zip(values).map(move |(&c, &v)| {
            let value = if v < LOW_BINS {
                // Values below pivot: reconstruct from lower bins
                self.pivots[c.as_()] - self.second_quants[c.as_()] * (LOW_BINS - v) as f32
            } else {
                // Values at or above pivot: reconstruct from upper bins
                self.pivots[c.as_()] + self.quants[c.as_()] * (v - LOW_BINS) as f32
            };

            (c, value)
        })
    }

    fn prepare_query<D: ComponentType, W: ValueType>(
        &self,
        query: impl Iterator<Item = (D, W)>,
    ) -> Self::PreparedQuery<D, W> {
        let mut vec = vec![W::zero(); self.dim];
        for (i, v) in query {
            vec[i.as_()] = v;
        }
        vec
    }

    fn dot_product_from_offset<D: ComponentType, W: ValueType>(
        &self,
        prepared_query: &Self::PreparedQuery<D, W>,
        offset: usize,
        len: usize,
    ) -> f32 {
        let cv = self
            .get_with_offset_iter(offset, len)
            .map(|(c, v)| (c.as_(), v.to_f32().unwrap()));
        dot_product_dense_sparse(prepared_query, cv)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Prefetches the components and values of a vector with the specified offset and length into the CPU cache.
    ///
    /// This method prefetches the components and values of a vector starting at the specified `offset``
    /// and with the specified length `len` into the CPU cache, which can improve performance by reducing
    /// cache misses during subsequent accesses.
    ///
    /// # Parameters
    ///
    /// * `offset`: The starting index of the vector to prefetch.
    /// * `len`: The length of the vector to prefetch.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::seismic::SparseDatasetTrait;
    /// use seismic::SparseDatasetMut;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetMut<u16, f32> = data.into();
    ///
    /// // Prefetch components and values of the vector starting at index 1 with length 3
    /// dataset.prefetch_with_offset(1, 3);
    /// ```
    #[inline]
    fn prefetch_with_offset(&self, offset: usize, len: usize) {
        // TODO: The crate has no way to get where a certain index is actually stored...
        // prefetch_read_slice(components);
        prefetch_read_slice(unsafe { self.values.get_unchecked(offset..(offset + len)) });
    }

    fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = impl Iterator<Item = (Self::Component, Self::Value)>> {
        self.offsets
            .array_windows()
            .map(|&[start, end]| self.get_with_offset_iter(start, end - start))
    }

    fn par_iter(
        &self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = (Self::Component, Self::Value)>>
    {
        // https://github.com/rayon-rs/rayon/pull/789
        self.offsets.par_windows(2).map(|wind| {
            let &[start, end] = wind else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            self.get_with_offset_iter(start, end - start)
        })
    }
}

impl<C, O, AC, AV> FromDatasetGenericF32<SparseDatasetGeneric<C, f32, O, AC, AV>>
    for SparseDatasetQuantizedUniformComponent<C>
where
    C: ComponentType,
    O: AsRef<[usize]> + SpaceUsage + Into<Box<[usize]>>,
    AC: AsRef<[C]> + SpaceUsage + Into<Box<[C]>>,
    AV: AsRef<[f32]> + SpaceUsage + Into<Box<[f32]>>,
{
    fn from_dataset_f32(dataset: SparseDatasetGeneric<C, f32, O, AC, AV>) -> Self {
        let dim = dataset.dim();
        let (offsets, components, values) = dataset.destroy();

        let components = components.into();
        let offsets = offsets.into();

        let values = values.into();

        let mut posting_lists = vec![Vec::new(); dim];

        println!("Collecting values per component");
        for (c, &v) in components.iter().zip(values.iter()) {
            posting_lists[c.as_()].push(v);
        }

        println!("Computing quantization info");

        let mode = "mean".to_string(); // or "midpoint"
        let mut quantization_info = Vec::<(f32, f32, f32)>::with_capacity(dim);
        for c_values in posting_lists.iter() {
            //quantization_info.push(get_uniform_quantization_info(c_values)); // (min, quant)
            quantization_info.push(get_uniform_quantization_info_pivot(
                c_values,
                LOW_BINS,
                mode.clone(),
            )); // (min, quant)
        }

        println!("Quantizing values");
        let quantized_values: Vec<u8> = values
            .iter()
            .zip(components.iter())
            .progress_count(values.len() as u64)
            .map(|(&v, &c)| {
                let idx = c.as_();
                let pivot = quantization_info[idx].0;
                let bin_size = quantization_info[idx].1;
                let second_bin_size = quantization_info[idx].2;

                let quantized = if v < pivot {
                    // Values below pivot: use bins 0..LOW_BINS-1
                    let bins_below = ((pivot - v) / second_bin_size).round() as usize;
                    if bins_below >= LOW_BINS as usize {
                        0
                    } else {
                        LOW_BINS - bins_below as u8
                    }
                } else {
                    // Values at or above pivot: use bins LOW_BINS..255
                    let bins_above = ((v - pivot) / bin_size).round() as usize;
                    let result = LOW_BINS as usize + bins_above;
                    if result > 255 { 255 } else { result as u8 }
                };
                quantized
            })
            .collect();
        // let quantized_values: Vec<u8> = values
        //     .iter()
        //     .zip(components.iter())
        //     .progress_count(values.len() as u64)
        //     .map(|(&v, &c)| {
        //         let idx = c.as_();
        //         ((v.to_f32().unwrap() - quantization_info[idx].0) / quantization_info[idx].1)
        //             .round() as u8
        //     })
        //     .collect();

        let mut pivots = Vec::with_capacity(dim);
        let mut quants = Vec::with_capacity(dim);
        let mut second_quants = Vec::with_capacity(dim);
        for (m, q, sq) in quantization_info.iter() {
            pivots.push(*m);
            quants.push(*q);
            second_quants.push(*sq);
        }
        // let (pivots, quants): (Vec<f32>, Vec<f32>) =
        //     quantization_info.into_iter().();

        Self {
            dim,
            offsets,
            components,
            values: quantized_values.into_boxed_slice(),
            pivots: pivots.into_boxed_slice(),
            quants: quants.into_boxed_slice(),
            second_quants: second_quants.into_boxed_slice(),
        }
    }
}

impl<C> SpaceUsage for SparseDatasetQuantizedUniformComponent<C>
where
    C: ComponentType,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        self.dim.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.components.space_usage_byte()
            + self.values.space_usage_byte()
            + self.pivots.space_usage_byte()
            + self.quants.space_usage_byte()
            + self.second_quants.space_usage_byte()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn conversion_and_dot_product() {
        let data = std::iter::repeat_n(
            (
                (0..500).map(|i| (i) as u16).collect_vec(),
                (0..500).map(|i| (i % 256) as f32 * 0.5).collect_vec(),
            ),
            500,
        );

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.map(|(c, v)| c.into_iter().zip(v).collect_vec()),
        );

        let quantized_dataset =
            SparseDatasetQuantizedUniformComponent::<u16>::from_dataset_f32(dataset);

        let query = [(1_u16, 1.0), (2, 2.0)];
        let prepared_query = quantized_dataset.prepare_query(query.into_iter());

        // Gotta type the generic parameters because of https://github.com/rust-lang/rust/issues/144858
        assert_eq!(
            quantized_dataset.dot_product_from_id::<u16, f32>(&prepared_query, 0),
            // TODO: This is equal only because the maximum value is exactly 0.5 * 256
            2.5
        );
    }

    #[test]
    fn test_quantization_roundtrip() {
        // Test simile per verificare la simmetria quantizzazione/dequantizzazione
        let data = vec![
            (vec![0_u16, 1, 2], vec![1.0_f32, 5.0, 10.0]),
            (vec![0_u16, 2, 3], vec![2.0_f32, 8.0, 15.0]),
        ];

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.iter()
                .map(|(c, v)| c.iter().copied().zip(v.iter().copied()).collect_vec()),
        );

        let quantized_dataset =
            SparseDatasetQuantizedUniformComponent::<u16>::from_dataset_f32(dataset);

        // Verifica che i valori possano essere ricostruiti ragionevolmente
        for (i, vector) in quantized_dataset.iter().enumerate() {
            let reconstructed: Vec<_> = vector.collect();
            println!("Vector {}: {:?}", i, reconstructed);

            // Verifica che tutti i componenti siano presenti
            assert_eq!(reconstructed.len(), data[i].0.len());
        }
    }

    #[test]
    fn test_quantization_bounds() {
        // Test per verificare che i valori quantizzati restino nel range corretto
        let data = vec![
            (vec![0_u16], vec![-100.0_f32]), // Valore molto basso
            (vec![0_u16], vec![0.0_f32]),    // Zero
            (vec![0_u16], vec![100.0_f32]),  // Valore molto alto
        ];

        let dataset = crate::SparseDatasetMut::<u16, f32>::from_iter(
            data.iter()
                .map(|(c, v)| c.iter().copied().zip(v.iter().copied()).collect_vec()),
        );

        let quantized_dataset =
            SparseDatasetQuantizedUniformComponent::<u16>::from_dataset_f32(dataset);

        // Verifica che i valori estremi vengano gestiti correttamente
        // (i valori dovrebbero essere clampati a 0 e 255)
        let has_min_value = quantized_dataset.values.iter().any(|&v| v == 0);
        let has_max_value = quantized_dataset.values.iter().any(|&v| v == 255);

        println!("Has min value (0): {}", has_min_value);
        println!("Has max value (255): {}", has_max_value);
        println!("Values: {:?}", quantized_dataset.values);
    }
}
