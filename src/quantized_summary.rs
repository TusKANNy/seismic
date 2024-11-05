use half::f16;

use serde::{Deserialize, Serialize};

use crate::{SpaceUsage, SparseDataset};

use qwt::SpaceUsage as QwtSpaceUsage;

use qwt::{DArray, SelectBin};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct QuantizedSummary {
    n_summaries: usize,
    d: usize,
    offsets: DArray<false>,
    summaries_ids: Box<[u16]>, // There cannot be more than 2^16 summaries
    values: Box<[u8]>,
    minimums: Box<[f32]>,
    quants: Box<[f32]>,
}

impl QuantizedSummary {
    const N_CLASSES: usize = 256; // we store quantized values in a u8. Number of classes cannot be more than 256

    pub fn space_usage_byte(&self) -> usize {
        SpaceUsage::space_usage_byte(&self.n_summaries)
            + SpaceUsage::space_usage_byte(&self.d)
            + QwtSpaceUsage::space_usage_byte(&self.offsets)
            + SpaceUsage::space_usage_byte(&self.summaries_ids)
            + SpaceUsage::space_usage_byte(&self.values)
            + SpaceUsage::space_usage_byte(&self.minimums)
            + SpaceUsage::space_usage_byte(&self.quants)
    }

    pub fn matmul_with_query(&self, query_components: &[u16], query_values: &[f32]) -> Vec<f32> {
        let mut accumulator = vec![0_f32; self.n_summaries];

        for (&qc, &qv) in query_components.iter().zip(query_values) {
            let current_offset = self.offsets.select1(qc as usize).unwrap() - qc as usize;
            let next_offset = self.offsets.select1((qc + 1) as usize).unwrap() - qc as usize - 1;

            if next_offset - current_offset == 0 {
                continue;
            }
            let current_summaries_ids = &self.summaries_ids[current_offset..next_offset];
            let current_values = &self.values[current_offset..next_offset];

            for (&s_id, &v) in current_summaries_ids.iter().zip(current_values) {
                let val = v as f32 * self.quants[s_id as usize] + self.minimums[s_id as usize];
                //accumulator[c as usize] += v as f32 * qv;
                //q_vs[c as usize] += qv;
                accumulator[s_id as usize] += val * qv;
            }

            // for i in 0..accumulator.len() {
            //     accumulator[i] = accumulator[i] * self.quants[i] + self.minimums[i] * q_vs[i];
            // }
        }

        accumulator
    }

    /// # Panics
    /// Panics if the number of summmaries is more than 2^16 (i.e., u16::MAX)
    pub fn new(dataset: SparseDataset<f16>, original_dim: usize) -> QuantizedSummary {
        // We need the original dim because the summaries for the current posting list
        // may not contain all the components. An alternative is to use an HashMap to map
        // the components

        assert!(
            dataset.len() <= u16::MAX as usize,
            "Number of summaries cannot be more than 2^16"
        );

        // TODO: if dim is big it may be better to use a HashMap to map the components to the summaries
        let mut inverted_pairs = Vec::with_capacity(original_dim);
        for _ in 0..original_dim {
            inverted_pairs.push(Vec::new());
        }

        let mut minimums = Vec::with_capacity(inverted_pairs.len());
        let mut quants = Vec::with_capacity(inverted_pairs.len());

        for (doc_id, (components, values)) in dataset.iter().enumerate() {
            let (minimum, quant, current_codes) = Self::quantize(values);

            minimums.push(minimum);
            quants.push(quant);

            for (&c, score) in components.iter().zip(current_codes) {
                inverted_pairs[c as usize].push((score, doc_id));
            }
        }

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len());
        let mut summaries_ids: Vec<u16> = Vec::with_capacity(dataset.nnz());
        let mut codes = Vec::with_capacity(dataset.nnz());

        offsets.push(0);
        for ip in inverted_pairs.iter() {
            codes.extend(ip.iter().map(|(s, _)| *s));
            summaries_ids.extend(ip.iter().map(|(_, id)| *id as u16));
            offsets.push(summaries_ids.len())
        }

        QuantizedSummary {
            n_summaries: dataset.len(),
            d: dataset.dim(),
            offsets: offsets
                .into_iter()
                .enumerate()
                .map(|(id, cur_offset)| cur_offset + id) // Add id to make a strictly increasing sequence
                .collect(),
            summaries_ids: summaries_ids.into_boxed_slice(),
            values: codes.into_boxed_slice(),
            minimums: minimums.into_boxed_slice(),
            quants: quants.into_boxed_slice(),
        }
    }

    #[inline]
    fn quantize(values: &[f16]) -> (f32, f32, Vec<u8>) {
        assert!(!values.is_empty());

        // Compute min and max values in the vector
        let (min, max) = values.iter().fold((values[0], values[0]), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });

        let (min, max) = (min.to_f32(), max.to_f32());

        // Quantization splits the range [min, max] into Self::N_CLASSES blocks of equal size
        // (max-m)/Self::N_CLASSES.
        // Exponential quantization could be possible as well.

        let mut quantized_values = Vec::with_capacity(values.len());
        let quant = (max - min) / (Self::N_CLASSES as f32);
        for &v in values {
            let q = ((f32::from(v) - min) / quant) as u8;
            quantized_values.push(q);
        }

        (min, quant, quantized_values)
    }
}
