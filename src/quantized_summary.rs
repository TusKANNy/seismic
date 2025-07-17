use serde::{Deserialize, Serialize};

use crate::{ComponentType, DataType, SpaceUsage, SparseDataset};

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

    #[must_use]
    pub fn distances_iter(&self, query_components: &[C], query_values: &[f32]) -> DistancesIter {
        DistancesIter::new(self, query_components, query_values)
    }

    #[inline]
    #[must_use]
    fn quantize<T: DataType>(values: &[T]) -> (f32, f32, Vec<u8>) {
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
    T: DataType,
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

        // TODO: decide based on EF space usage
        let mut component_ids: Option<Vec<C>> = if false { Some(Vec::new()) } else { None };

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
            let current_offset = summaries.offsets.select(qc.as_()).unwrap();
            let next_offset = summaries.offsets.select(qc.as_() + 1).unwrap();

            if next_offset - current_offset == 0 {
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
