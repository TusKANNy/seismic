use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{ComponentType, SpaceUsage, SparseDataset, ValueType};

use rustc_hash::FxHashMap;

use toolkit::BitFieldBoxed;
use toolkit::EliasFano;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct QuantizedSummary<C: ComponentType> {
    n_summaries: usize,
    dim: usize,
    component_ids: Option<Box<[C]>>,
    offsets: EliasFano,
    summaries_ids: BitFieldBoxed,
    values: Box<[u8]>,
    minimums: Box<[f32]>,
    quants: Box<[f32]>,
}

use mem_dbg::{MemSize, SizeFlags};
impl SpaceUsage for BitFieldBoxed {
    fn space_usage_byte(&self) -> usize {
        self.mem_size(SizeFlags::empty())
    }
}

impl SpaceUsage for EliasFano {
    fn space_usage_byte(&self) -> usize {
        self.mem_size(SizeFlags::empty())
    }
}

impl<C: ComponentType> SpaceUsage for QuantizedSummary<C> {
    fn space_usage_byte(&self) -> usize {
        let component_ids_size = if let Some(ref component_ids) = self.component_ids {
            SpaceUsage::space_usage_byte(component_ids)
        } else {
            0_usize
        };

        component_ids_size
            + SpaceUsage::space_usage_byte(&self.n_summaries)
            + SpaceUsage::space_usage_byte(&self.dim)
            + SpaceUsage::space_usage_byte(&self.offsets)
            + SpaceUsage::space_usage_byte(&self.summaries_ids)
            + SpaceUsage::space_usage_byte(&self.values)
            + SpaceUsage::space_usage_byte(&self.minimums)
            + SpaceUsage::space_usage_byte(&self.quants)
    }
}

impl<C: ComponentType> QuantizedSummary<C> {
    const N_CLASSES: usize = 256; // we store quantized values in a u8. Number of classes cannot be more than 256

    /// Calculate space usage for sparse offset strategy (with component_ids)
    fn estimate_sparse_space(num_components: usize, max_offset: usize) -> usize {
        // Space for component_ids: number of components * size of component type C in bits
        let component_ids_space = num_components * std::mem::size_of::<C>() * 8;

        let ef_space = EliasFano::estimate_space_bits(max_offset + 1, num_components);

        component_ids_space + ef_space
    }

    /// Calculate space usage for dense offset strategy (without component_ids)
    fn estimate_dense_space(d: usize, max_offset: usize) -> usize {
        // Space for Elias-Fano offsets with full dimension d
        EliasFano::estimate_space_bits(max_offset + 1 + d, d)
    }

    pub fn distances(&self, query_components: &[C], query_values: &[f32]) -> Vec<f32> {
        let mut accumulator = vec![0_f32; self.n_summaries];

        if let Some(component_ids) = self.component_ids.as_ref() {
            let mut i = 0; // index for component_ids
            let mut j = 0; // index for query_components

            while i < component_ids.len() && j < query_components.len() {
                // TODO: dont do casting here once both are of type C::ComponentType
                let comp_id = component_ids[i].as_();
                let query_comp = query_components[j].as_();

                if comp_id == query_comp {
                    // Found matching component, process it
                    let qv = query_values[j];
                    let current_offset = self.offsets.select(i).unwrap();
                    let next_offset = self.offsets.select(i + 1).unwrap();

                    // let current_summaries_ids =
                    //     &summaries.summaries_ids[current_offset..next_offset];
                    let current_values =
                        unsafe { self.values.get_unchecked(current_offset..next_offset) };

                    for (pos, &v) in (current_offset..next_offset).zip(current_values) {
                        let s_id = unsafe { self.summaries_ids.get_unchecked(pos) as usize };
                        let dequantized_val = unsafe {
                            v as f32 * self.quants.get_unchecked(s_id)
                                + self.minimums.get_unchecked(s_id)
                        };
                        *unsafe { accumulator.get_unchecked_mut(s_id) } += dequantized_val * qv;
                    }

                    i += 1;
                    j += 1;
                } else if comp_id < query_comp {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        } else {
            // If there are no component_ids, we use the query_components directly
            // This is the case when we have a dense offset vector

            for (&qc, &qv) in query_components
                .iter()
                .zip(query_values)
                .take_while(|(qc, _)| (qc.as_()) < self.dim)
            {
                let current_offset = self.offsets.select(qc.as_()).unwrap() - qc.as_();
                let next_offset = self.offsets.select(qc.as_() + 1).unwrap() - (qc.as_() + 1);

                // let current_summaries_ids = unsafe {self.summaries_ids.get_unchecked(current_offset..next_offset)};
                let current_values =
                    unsafe { self.values.get_unchecked(current_offset..next_offset) };

                for (pos, &v) in (current_offset..next_offset).zip(current_values) {
                    let s_id = unsafe { self.summaries_ids.get_unchecked(pos) as usize };
                    let dequantized_val = unsafe {
                        v as f32 * self.quants.get_unchecked(s_id)
                            + self.minimums.get_unchecked(s_id)
                    };
                    *unsafe { accumulator.get_unchecked_mut(s_id) } += dequantized_val * qv;
                }

                // for i in 0..accumulator.len() {
                //     accumulator[i] = accumulator[i] * self.quants[i] + self.minimums[i] * q_vs[i];
                // }
            }
        }

        accumulator
    }

    fn quantize<T: ValueType>(values: &[T]) -> (f32, f32, Vec<u8>) {
        assert!(!values.is_empty());

        // Compute min and max values in the vector
        let (min, max) = values
            .iter()
            .minmax_by(|a, b| a.partial_cmp(b).unwrap())
            .into_option()
            .unwrap();

        let (min, max) = (min.to_f32().unwrap(), max.to_f32().unwrap());

        // Quantization splits the range [min, max] into n_classes blocks of equal size (max-min)/n_clasess.
        // Max value is likely going to be n_classes - 1, due to rounding errors.
        // (Exponential quantization could be possible as well.)
        let quant = (max - min) / (Self::N_CLASSES as f32);
        let quantized_values = values
            .iter()
            .map(|&v| ((v.to_f32().unwrap() - min) / quant) as u8)
            .collect();

        (min, quant, quantized_values)
    }

    /// Print detailed space usage breakdown for this QuantizedSummary
    pub fn print_space_usage(&self) {
        // Calcolo la dimensione in byte di ogni campo
        let component_ids_size = if let Some(ref component_ids) = self.component_ids {
            SpaceUsage::space_usage_byte(component_ids)
        } else {
            0_usize
        };

        let n_summaries_size = SpaceUsage::space_usage_byte(&self.n_summaries);
        let d_size = SpaceUsage::space_usage_byte(&self.dim);
        let offsets_size = SpaceUsage::space_usage_byte(&self.offsets);
        let summaries_ids_size = SpaceUsage::space_usage_byte(&self.summaries_ids);
        let values_size = SpaceUsage::space_usage_byte(&self.values);
        let minimums_size = SpaceUsage::space_usage_byte(&self.minimums);
        let quants_size = SpaceUsage::space_usage_byte(&self.quants);

        // Calcolo il totale
        let total_size = component_ids_size
            + n_summaries_size
            + d_size
            + offsets_size
            + summaries_ids_size
            + values_size
            + minimums_size
            + quants_size;

        // Stampo il report dettagliato
        println!("\n=== QuantizedSummary Space Usage ===");
        println!(
            "Total size: {} bytes ({:.2} MB)",
            total_size,
            total_size as f64 / 1_048_576.0
        );
        println!("n_summaries: {}", self.n_summaries);
        println!("d (dimensions): {}", self.dim);

        if let Some(ref component_ids) = self.component_ids {
            println!(
                "Strategy: SPARSE (component_ids present, {} components)",
                component_ids.len()
            );
        } else {
            println!("Strategy: DENSE (no component_ids)");
        }

        println!("\n--- Component Breakdown ---");

        if component_ids_size > 0 {
            println!(
                "component_ids:   {:>12} bytes ({:>6.2}%)",
                component_ids_size,
                component_ids_size as f64 / total_size as f64 * 100.0
            );
        } else {
            println!(
                "component_ids:   {:>12} bytes ({:>6.2}%) [DENSE strategy]",
                0, 0.0
            );
        }

        println!(
            "n_summaries:     {:>12} bytes ({:>6.2}%)",
            n_summaries_size,
            n_summaries_size as f64 / total_size as f64 * 100.0
        );

        println!(
            "d:               {:>12} bytes ({:>6.2}%)",
            d_size,
            d_size as f64 / total_size as f64 * 100.0
        );

        println!(
            "offsets (EF):    {:>12} bytes ({:>6.2}%) [{} offsets]",
            offsets_size,
            offsets_size as f64 / total_size as f64 * 100.0,
            self.offsets.len()
        );

        println!(
            "summaries_ids:   {:>12} bytes ({:>6.2}%) [{} summaries {} distinct]",
            summaries_ids_size,
            summaries_ids_size as f64 / total_size as f64 * 100.0,
            self.summaries_ids.len(),
            self.n_summaries
        );

        println!(
            "values:          {:>12} bytes ({:>6.2}%) [{} values]",
            values_size,
            values_size as f64 / total_size as f64 * 100.0,
            self.values.len()
        );

        println!(
            "minimums:        {:>12} bytes ({:>6.2}%)",
            minimums_size,
            minimums_size as f64 / total_size as f64 * 100.0
        );

        println!(
            "quants:          {:>12} bytes ({:>6.2}%)",
            quants_size,
            quants_size as f64 / total_size as f64 * 100.0
        );

        println!("=====================================\n");
    }
}

impl<C, T> From<SparseDataset<C, T>> for QuantizedSummary<C>
where
    C: ComponentType,
    T: ValueType,
{
    /// # Panics
    /// Panics if the number of summmaries is more than 2^16 (i.e., u16::MAX)
    fn from(dataset: SparseDataset<C, T>) -> QuantizedSummary<C> {
        Self::from(&dataset)
    }
}

impl<C, T> From<&SparseDataset<C, T>> for QuantizedSummary<C>
where
    C: ComponentType,
    T: ValueType,
{
    /// # Panics
    /// Panics if the number of summmaries is more than 2^16 (i.e., u16::MAX)
    fn from(dataset: &SparseDataset<C, T>) -> QuantizedSummary<C> {
        assert!(
            dataset.len() <= u16::MAX as usize,
            "Number of summaries cannot be more than 2^16"
        );

        let mut inverted_pairs: FxHashMap<C, Vec<(u8, usize)>> =
            FxHashMap::with_capacity_and_hasher(1 << 16, Default::default());

        let mut minimums = Vec::with_capacity(inverted_pairs.len());
        let mut quants = Vec::with_capacity(inverted_pairs.len());

        for (doc_id, (components, values)) in dataset.iter().enumerate() {
            let (minimum, quant, current_codes) = Self::quantize(values);

            minimums.push(minimum);
            quants.push(quant);

            for (&c, score) in components.iter().zip(current_codes) {
                inverted_pairs.entry(c).or_default().push((score, doc_id));
            }
        }

        // Sort inverted pairs by component id
        let mut inverted_pairs: Vec<(C, Vec<(u8, usize)>)> = inverted_pairs.into_iter().collect();
        inverted_pairs.sort_by_key(|(component, _)| *component);

        // Calculate spaces for both strategies to choose the most efficient one
        let num_non_empty_components = inverted_pairs.len();
        let total_postings = inverted_pairs
            .iter()
            .map(|(_, list)| list.len())
            .sum::<usize>();

        let sparse_space = Self::estimate_sparse_space(num_non_empty_components, total_postings);
        let dense_space = Self::estimate_dense_space(dataset.dim(), dataset.dim() + total_postings);

        // Choose sparse strategy if it uses less space
        let use_sparse_strategy = sparse_space < dense_space;

        // println!("Sparse space: {}, Dense space: {}", sparse_space, dense_space);
        // println!("Using sparse strategy: {}", use_sparse_strategy);

        let mut component_ids: Option<Vec<C>> = if use_sparse_strategy {
            Some(Vec::new())
        } else {
            None
        };

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len());
        let mut summaries_ids: Vec<u64> = Vec::with_capacity(dataset.nnz());
        let mut codes = Vec::with_capacity(dataset.nnz());

        offsets.push(0);
        let mut prev_component = 0_usize;

        for (c, ip) in inverted_pairs.iter() {
            codes.extend(ip.iter().map(|(s, _)| *s));
            summaries_ids.extend(ip.iter().map(|(_, id)| *id as u64));
            if let Some(ref mut component_ids) = component_ids {
                // Sparse offset strategy: Store only occuring components
                component_ids.push(C::from_usize(c.as_()).unwrap());
                offsets.push(summaries_ids.len());
            } else {
                // Dense offset strategy stores all components.
                // So, we need to fill the gaps on the offsets for non existing components.
                let last_offset = *offsets.last().unwrap();
                for _ in prev_component..c.as_() {
                    offsets.push(last_offset);
                }
                offsets.push(summaries_ids.len());
                prev_component = c.as_() + 1;
            }
        }

        // In case of dense strategy, we want to make the offsets a strictly increasing sequence.
        // This is done by adding the index to the offset.
        // This is not needed for sparse strategy, since offsets are already strictly increasing.
        // Thie use of a strictly increasing sequence may be more space efficient for Elias-Fano.

        if !use_sparse_strategy {
            for (i, o) in offsets.iter_mut().enumerate() {
                *o += i;
            }
        }

        Self {
            n_summaries: dataset.len(),
            dim: dataset.dim(),
            component_ids: component_ids.map(|c| c.into_boxed_slice()),
            offsets: EliasFano::from(&offsets),
            summaries_ids: BitFieldBoxed::from(summaries_ids),
            values: codes.into_boxed_slice(),
            minimums: minimums.into_boxed_slice(),
            quants: quants.into_boxed_slice(),
        }

        // me.print_space_usage();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseDatasetMut;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    fn generate_random_sparse_dataset<C>(
        seed: u64,
        n_vecs: usize,
        dim: usize,
        min_nnz: usize,
        max_nnz: usize,
        value: f32,
    ) -> SparseDataset<C, f32>
    where
        C: ComponentType + std::convert::TryFrom<usize>,
        <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut dataset = SparseDatasetMut::new();

        for _ in 0..n_vecs {
            let nnz = rng.random_range(min_nnz..=max_nnz);
            let mut components: Vec<usize> = (0..dim).collect();
            components.shuffle(&mut rng);
            components.truncate(nnz);
            components.sort_unstable();

            let components: Vec<C> = components
                .into_iter()
                .map(|x| C::try_from(x).unwrap())
                .collect();
            let values = vec![value; nnz];
            dataset.push(&components, &values);
        }

        dataset.into()
    }

    fn generate_random_queries<C>(
        seed: u64,
        n_queries: usize,
        dim: usize,
        min_nnz: usize,
        max_nnz: usize,
    ) -> SparseDatasetMut<C, f32>
    where
        C: ComponentType + std::convert::TryFrom<usize>,
        <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut dataset = SparseDatasetMut::new();

        for _ in 0..n_queries {
            let nnz = rng.random_range(min_nnz..=max_nnz);
            let mut components: Vec<usize> = (0..dim).collect();
            components.shuffle(&mut rng);
            components.truncate(nnz);
            components.sort_unstable();

            let components: Vec<C> = components
                .into_iter()
                .map(|x| C::try_from(x).unwrap())
                .collect();
            let values: Vec<f32> = (0..nnz).map(|_| rng.random::<f32>()).collect();
            dataset.push(&components, &values);
        }

        dataset
    }

    fn compute_inner_product<C: ComponentType>(
        query_components: &[C],
        query_values: &[f32],
        vec_components: &[C],
        vec_values: &[f32],
    ) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut result = 0.0;

        while i < query_components.len() && j < vec_components.len() {
            let qc = query_components[i].as_();
            let vc = vec_components[j].as_();

            if qc == vc {
                result += query_values[i] * vec_values[j];
                i += 1;
                j += 1;
            } else if qc < vc {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }

    #[test]
    fn test_distances_iter() {
        let seed = 142;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate random dimensions within specified ranges
        let n_vecs = rng.random_range(50..=100);
        let dim = rng.random_range(100_000..=140_000);
        let min_nnz = 300;
        let max_nnz = 500;

        println!(
            "Test parameters: n_vecs={}, dim={}, min_nnz={}, max_nnz={}, seed={}",
            n_vecs, dim, min_nnz, max_nnz, seed
        );

        // Generate dataset with all values set to 1.0
        let dataset: SparseDataset<u32, f32> =
            generate_random_sparse_dataset(seed, n_vecs, dim, min_nnz, max_nnz, 1.0);

        // Generate 100 random queries
        let mut queries = generate_random_queries(
            seed + 1, // Different seed for queries
            100,
            dim,
            min_nnz,
            max_nnz,
        );

        // Also add dataset itselft to the queries
        for (c, v) in dataset.iter() {
            queries.push(c, v);
        }

        // Create quantized summary
        let summary = QuantizedSummary::from(&dataset);

        // For each query, compare distances
        for (query_id, (query_components, query_values)) in queries.iter().enumerate() {
            // Get distances using DistancesIter
            let distances: Vec<f32> = summary.distances(query_components, query_values);
            assert_eq!(distances.len(), dataset.len());

            // Compute distances explicitly
            let mut expected_distances = Vec::with_capacity(dataset.len());
            for (vec_components, vec_values) in dataset.iter() {
                let distance = compute_inner_product(
                    query_components,
                    query_values,
                    vec_components,
                    vec_values,
                );
                expected_distances.push(distance);
            }

            // Compare results
            for (i, (got, expected)) in distances.iter().zip(expected_distances.iter()).enumerate()
            {
                assert!(
                    (got - expected).abs() < 1e-5,
                    "Distance mismatch for query {} vector {}: got {}, expected {} (diff: {})",
                    query_id,
                    i,
                    got,
                    expected,
                    (got - expected).abs()
                );
            }
        }
    }
}
