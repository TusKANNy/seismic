use serde::{Deserialize, Serialize};

use crate::{ComponentType, SpaceUsage, SparseDataset, ValueType};

use crate::elias_fano::EliasFano;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct QuantizedSummary<C: ComponentType> {
    n_summaries: usize,
    d: usize,
    component_ids: Option<Box<[C]>>,
    offsets: EliasFano,
    summaries_ids: Box<[u16]>, // There cannot be more than 2^16 summaries
    values: Box<[u8]>,
    minimums: Box<[f32]>,
    quants: Box<[f32]>,
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
            + SpaceUsage::space_usage_byte(&self.d)
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

        // Space for Elias-Fano offsets using the correct formula
        let ef_space = if num_components > 0 && max_offset > 0 {
            EliasFano::estimate_space_bits(max_offset + 1, num_components)
        } else {
            0
        };

        component_ids_space + ef_space
    }

    /// Calculate space usage for dense offset strategy (without component_ids)
    fn estimate_dense_space(d: usize, max_offset: usize) -> usize {
        // Space for Elias-Fano offsets with full dimension d
        if d > 0 && max_offset > 0 {
            EliasFano::estimate_space_bits(max_offset + 1, d)
        } else {
            0
        }
    }

    #[must_use]
    pub fn distances_iter(&self, query_components: &[C], query_values: &[f32]) -> DistancesIter {
        DistancesIter::new(self, query_components, query_values)
    }

    #[inline]
    #[must_use]
    fn quantize<T: ValueType>(values: &[T]) -> (f32, f32, Vec<u8>) {
        assert!(!values.is_empty());

        // Compute min and max values in the vector
        let (min, max) = values.iter().fold((values[0], values[0]), |acc, &v| {
            (
                if acc.0 < v { acc.0 } else { v },
                if acc.1 > v { acc.1 } else { v },
            )
        });

        let (min, max) = (min.to_f32().unwrap(), max.to_f32().unwrap());

        // Quantization splits the range [min, max] into Self::N_CLASSES blocks of equal size
        // (max-m)/Self::N_CLASSES.
        // Exponential quantization could be possible as well.

        let mut quantized_values = Vec::with_capacity(values.len());
        let quant = (max - min) / (Self::N_CLASSES as f32);
        for &v in values {
            let q = ((v.to_f32().unwrap() - min) / quant) as u8;
            quantized_values.push(q);
        }

        (min, quant, quantized_values)
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
        let dense_space = Self::estimate_dense_space(dataset.dim(), total_postings);

        // Choose sparse strategy if it uses less space
        let use_sparse_strategy = sparse_space < dense_space;

        let mut component_ids: Option<Vec<C>> = if use_sparse_strategy {
            Some(Vec::new())
        } else {
            None
        };

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len());
        let mut summaries_ids: Vec<u16> = Vec::with_capacity(dataset.nnz());
        let mut codes = Vec::with_capacity(dataset.nnz());

        offsets.push(0);
        let mut prev_component = 0_usize;

        for (c, ip) in inverted_pairs.iter() {
            codes.extend(ip.iter().map(|(s, _)| *s));
            summaries_ids.extend(ip.iter().map(|(_, id)| *id as u16));
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
            d: dataset.dim(),
            component_ids: if let Some(component_ids) = component_ids {
                Some(component_ids.into_boxed_slice())
            } else {
                None
            },
            offsets: EliasFano::from(&offsets),
            summaries_ids: summaries_ids.into_boxed_slice(),
            values: codes.into_boxed_slice(),
            minimums: minimums.into_boxed_slice(),
            quants: quants.into_boxed_slice(),
        }
    }
}

pub struct DistancesIter {
    current: usize,
    distances: Vec<f32>,
}

impl DistancesIter {
    fn new<C>(summaries: &QuantizedSummary<C>, query_components: &[C], query_values: &[f32]) -> Self
    where
        C: ComponentType,
    {
        let mut accumulator = vec![0_f32; summaries.n_summaries];

        if let Some(component_ids) = &summaries.component_ids {
            let mut i = 0; // index for component_ids
            let mut j = 0; // index for query_components

            while i < component_ids.len() && j < query_components.len() {
                // TODO: dont do casting here once both are of type C::ComponentType
                let comp_id = component_ids[i].as_();
                let query_comp = query_components[j].as_();

                if comp_id == query_comp {
                    // Found matching component, process it
                    let qv = query_values[j];
                    let current_offset = summaries.offsets.select(i).unwrap();
                    let next_offset = summaries.offsets.select(i + 1).unwrap();

                    let current_summaries_ids =
                        &summaries.summaries_ids[current_offset..next_offset];
                    let current_values = &summaries.values[current_offset..next_offset];

                    for (&s_id, &v) in current_summaries_ids.iter().zip(current_values) {
                        let val = v as f32 * summaries.quants[s_id as usize]
                            + summaries.minimums[s_id as usize];
                        accumulator[s_id as usize] += val * qv;
                    }

                    i += 1;
                    j += 1;
                } else if comp_id < query_comp {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            return Self {
                current: 0,
                distances: accumulator,
            };
        }
        // If there are no component_ids, we use the query_components directly
        // This is the case when we have a dense offset vector

        for (&qc, &qv) in query_components.iter().zip(query_values) {
            if qc.as_() >= summaries.d {
                break;
            }
            let current_offset = summaries.offsets.select(qc.as_()).unwrap() - qc.as_();
            let next_offset = summaries.offsets.select(qc.as_() + 1).unwrap() - (qc.as_() + 1);

            if current_offset == next_offset {
                continue;
            }
            let current_summaries_ids = &summaries.summaries_ids[current_offset..next_offset];
            let current_values = &summaries.values[current_offset..next_offset];

            for (&s_id, &v) in current_summaries_ids.iter().zip(current_values) {
                let val =
                    v as f32 * summaries.quants[s_id as usize] + summaries.minimums[s_id as usize];
                //accumulator[c as usize] += v as f32 * qv;
                //q_vs[c as usize] += qv;
                accumulator[s_id as usize] += val * qv;
            }

            // for i in 0..accumulator.len() {
            //     accumulator[i] = accumulator[i] * self.quants[i] + self.minimums[i] * q_vs[i];
            // }
        }

        Self {
            current: 0,
            distances: accumulator,
        }
    }
}

impl Iterator for DistancesIter {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.distances.len() {
            let current = self.current;
            self.current += 1;
            Some(self.distances[current])
        } else {
            None
        }
    }
}
