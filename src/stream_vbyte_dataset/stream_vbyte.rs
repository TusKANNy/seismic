use std::simd::StdFloat;
use std::simd::num::{SimdFloat, SimdUint};
use std::{
    mem::transmute_copy,
    simd::{Mask, Simd},
};

use bytemuck::try_cast_slice;
use num_traits::ToPrimitive;
use rusty_perm::*;

use crate::FixedU8Q;
use crate::stream_vbyte_dataset::swizzle::swizzle;

const N: usize = u8::BITS as usize;
pub(super) const MASKS: [Simd<u8, { N * 2 }>; 256] = generate_masks_u16();

const fn generate_masks_u16() -> [Simd<u8, { N * 2 }>; 256] {
    let mut masks = [Simd::splat(0); 256];
    let mut i: usize = 0;
    while i < masks.len() {
        let mask = generate_mask_u16(i as u8);
        masks[i] = mask;
        i += 1;
    }
    masks
}

const fn generate_mask_u16(i: u8) -> Simd<u8, { N * 2 }> {
    let mut mask = [u8::MAX; N * 2];
    let mut j = 0;
    let mut scroll = 0;
    while j < 8 {
        let bytes = if (i & (0b1000_0000 >> j)) > 0 {
            // If two bytes, they are already in the correct endianness
            let n = [scroll, scroll + 1];
            scroll += 2;
            n
        } else {
            // If one byte, swizzle to the correct endianness
            let n = u16::from_be_bytes([u8::MAX, scroll]).to_ne_bytes();
            scroll += 1;
            n
        };
        mask[j * 2] = bytes[0];
        mask[j * 2 + 1] = bytes[1];

        j += 1;
    }

    Simd::from_array(mask)
}

fn simd_prefix_sum<const N: usize>(mut n: Simd<u16, N>) -> Simd<u16, N>
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
{
    // I'd use a for loop, but the const argument prevents doing that...
    // God I wish there was an easier way to do this
    if N > 1 {
        n += n.shift_elements_right::<1>(0);
    }
    if N > 2 {
        n += n.shift_elements_right::<2>(0);
    }
    if N > 4 {
        n += n.shift_elements_right::<4>(0);
    }
    // TODO: N more than 8
    n
}

fn simd_fixedu8_to_f32<const N: usize>(f: Simd<u8, N>) -> Simd<f32, N>
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
{
    let converted_f32 = f.cast();
    // This is *so* hardcoded
    let mult = Simd::splat(1.0 / (1 << crate::FixedU8Q::FRAC_NBITS) as f32);
    converted_f32 * mult
}

#[derive(Clone)]
pub struct StreamVbyte<'a> {
    values: &'a [Simd<u8, N>],
    bytes_remaining: &'a [u16],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> StreamVbyte<'a> {
    // We have got:
    // - The compressed bytes, which are unaligned by design, and is "unpredictable" to know where they end
    // - The idx_lens, which being an array of bytes read one at a time, doesn't have alignment problems
    // - The values, which being read in SIMD, should preferably aligned
    // - The remaining stuff, only the bytes of the components (always being u16) need alignment
    // Which is why the values (size of u8x8) come first, followed by the remaining bytes (size of u16), and everything else comes last.
    pub unsafe fn from_unchecked(slice: &'a [usize], n_elems: usize) -> Self {
        unsafe {
            let slice = try_cast_slice::<usize, u8>(slice).unwrap_unchecked();
            let n_packs = n_elems / N;
            let n_remaining = n_elems % N;

            let values_end = n_packs * size_of::<Simd<u8, N>>();
            let values = try_cast_slice(slice.get_unchecked(..values_end)).unwrap_unchecked();

            // These `next_multiple_of` are no-ops, I just want to express the importance of alignment.
            let bytes_remaining_start = values_end.next_multiple_of(align_of::<u16>());
            let bytes_remaining_end = bytes_remaining_start + n_remaining * size_of::<u16>();
            let bytes_remaining =
                try_cast_slice(slice.get_unchecked(bytes_remaining_start..bytes_remaining_end))
                    .unwrap_unchecked();

            let values_remaining_start = bytes_remaining_end.next_multiple_of(align_of::<u8>());
            let values_remaining_end = values_remaining_start + n_remaining * size_of::<u8>();
            let values_remaining =
                try_cast_slice(slice.get_unchecked(values_remaining_start..values_remaining_end))
                    .unwrap_unchecked();

            let idx_lens_start = values_remaining_end.next_multiple_of(align_of::<u8>());
            let idx_lens_end = idx_lens_start + n_packs * size_of::<u8>();
            let idx_lens = try_cast_slice(slice.get_unchecked(idx_lens_start..idx_lens_end))
                .unwrap_unchecked();

            // Next part is unaligned by design
            let bytes_start = idx_lens_end;
            // TODO: this unbounded length is "safe": *after the swizzle*, the only read values are of this posting.
            // But *before the swizzle*, for the last document, some values may be out of the slice's bounds.
            let bytes = try_cast_slice(slice.get_unchecked(bytes_start..)).unwrap_unchecked();

            Self {
                values,
                bytes_remaining,
                values_remaining,
                idx_lens,
                bytes,
            }
        }
    }

    pub fn push_posting(vec: &mut Vec<u8>, converted_components: &mut [u16], values: &mut [u8]) {
        assert_eq!(converted_components.len(), values.len());

        let permutation = PermD::from_sort(&*converted_components);
        permutation.apply(values).unwrap();
        permutation.apply(converted_components).unwrap();

        for i in (1..converted_components.len()).rev() {
            converted_components[i] -= converted_components[i - 1];
        }

        let n_chunked = converted_components.len() - converted_components.len() % N;
        let (components_chunked, components_remaining) =
            unsafe { converted_components.split_at_unchecked(n_chunked) };
        let (values_chunked, values_remaining) = unsafe { values.split_at_unchecked(n_chunked) };

        let bitvec: Vec<u8> = components_chunked
            .chunks_exact(N)
            .map(|chunk| {
                let mut byte = 0;
                for (i, b) in chunk.iter().map(|&n| n > u8::MAX as u16).enumerate() {
                    byte |= (b as u8) << (N - i - 1)
                }
                byte
            })
            .collect();

        unsafe {
            vec.extend_from_slice(try_cast_slice(values_chunked).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u16>()), 0);
            vec.extend_from_slice(try_cast_slice(components_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(values_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(&bitvec).unwrap_unchecked());

            for &c in components_chunked.iter() {
                if c > u8::MAX as u16 {
                    vec.extend_from_slice(&c.to_ne_bytes());
                } else {
                    vec.extend_from_slice(&[c as u8]);
                }
            }
            // The vector is aligned to a usize
            const { assert!(size_of::<usize>().is_multiple_of(size_of::<Simd<u8, N>>())) };
            vec.resize(vec.len().next_multiple_of(size_of::<Simd<u8, N>>()), 0);
        }
    }

    fn iter_raw(self) -> impl ExactSizeIterator<Item = (Simd<u16, N>, Simd<u8, N>)> {
        let mut total_scroll = 0;
        self.idx_lens
            .iter()
            .map(move |idx_len| {
                let bytes = unsafe {
                    self.bytes
                        .as_ptr()
                        .add(total_scroll)
                        .cast::<Simd<u8, 16>>()
                        .read_unaligned()
                };
                let mask = MASKS[*idx_len as usize];
                total_scroll += 8 + idx_len.count_ones() as usize;
                let result = swizzle(bytes, mask);
                unsafe { transmute_copy(&result) }
            })
            .zip(self.values.iter().cloned())
    }

    pub fn iter(self) -> impl Iterator<Item = (u16, FixedU8Q)> {
        let bytes_remaining = self.bytes_remaining;
        let values_remaining = self.values_remaining;
        gen move {
            let mut last_component = 0;
            for (c, v) in self.iter_raw() {
                let c_prefixed = simd_prefix_sum(c);
                let c_prefixed_previous = c_prefixed + Simd::splat(last_component);
                last_component = *c_prefixed_previous.to_array().last().unwrap();

                for (c, v) in c_prefixed_previous
                    .to_array()
                    .into_iter()
                    .zip(v.to_array().into_iter().map(FixedU8Q::from_bits))
                {
                    yield (c, v);
                }
            }
            for (c, v) in bytes_remaining.iter().zip(values_remaining.iter()) {
                last_component += c;
                yield (last_component, FixedU8Q::from_bits(*v));
            }
        }
    }

    pub fn dot_product(&self, mut query: &[f32]) -> f32 {
        let mut result = Simd::<f32, 8>::splat(0.0);
        // This ugly clone is optimized away
        for (components, values) in self.clone().iter_raw() {
            let components = simd_prefix_sum(components);
            let values = simd_fixedu8_to_f32(values);
            let query_values = unsafe {
                Simd::gather_select_unchecked(
                    query,
                    Mask::splat(true),
                    components.cast(),
                    Simd::splat(0.0),
                )
            };

            result = values.mul_add(query_values, result);
            let last_component = *components.to_array().last().unwrap();
            // New starting point for the gather
            query = unsafe { query.split_at_unchecked(last_component as usize).1 };
        }
        let simd_result = result.reduce_sum();

        let remaining_result = self
            .bytes_remaining
            .iter()
            .zip(self.values_remaining.iter())
            .scan(0, move |acc, (&c, &v)| {
                *acc += c;
                let query_value = FixedU8Q::from_bits(v).to_f32().unwrap();

                Some(unsafe {
                    query
                        .get_unchecked(*acc as usize)
                        .algebraic_mul(query_value)
                })
            })
            .fold(0f32, |acc, x| acc.algebraic_add(x));

        simd_result.algebraic_add(remaining_result)
    }
}
