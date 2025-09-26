#![feature(gen_blocks)]
#![feature(yield_expr)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use seismic::{ComponentType, ValueType};
use seismic::{FixedU16Q, FromDatasetGenericF32, SparseDatasetMut};
use std::hint::assert_unchecked;
use toolkit::{SVBEncodable, StreamVByteRandomAccess};

// Implementazioni diverse da testare
struct BenchDataset<C, V>
where
    C: ComponentType + SVBEncodable,
    V: ValueType,
{
    compressed_dataset: seismic::compressed_dataset::SparseDatasetCompressed<C, V>,
}

impl<C, V> BenchDataset<C, V>
where
    C: ComponentType + SVBEncodable + num_traits::ops::bytes::FromBytes,
    <C as num_traits::ops::bytes::FromBytes>::Bytes: Sized + Default,
    V: ValueType,
{
    // Implementazione originale con generator
    #[inline]
    fn get_with_offset_iter_generator(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (C, V)> {
        unsafe { assert_unchecked(len > 0) };
        let mut reader = self.compressed_dataset.components.iter_range(offset, len);
        gen move {
            for i in offset..(offset + len) {
                unsafe {
                    yield (
                        reader.next().unwrap(),
                        *self.compressed_dataset.values.get_unchecked(i),
                    )
                }
            }
        }
    }

    // Implementazione con slice e map
    #[inline]
    fn get_with_offset_iter_slice_map(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (C, V)> {
        unsafe { assert_unchecked(len > 0) };
        let mut reader = self.compressed_dataset.components.iter_range(offset, len);
        let values_slice = unsafe {
            self.compressed_dataset
                .values
                .get_unchecked(offset..offset + len)
        };

        values_slice
            .iter()
            .map(move |&value| (reader.next().unwrap(), value))
    }

    // Implementazione con zip
    #[inline]
    fn get_with_offset_iter_zip(&self, offset: usize, len: usize) -> impl Iterator<Item = (C, V)> {
        unsafe { assert_unchecked(len > 0) };
        let reader = self.compressed_dataset.components.iter_range(offset, len);
        let values_slice = unsafe {
            self.compressed_dataset
                .values
                .get_unchecked(offset..offset + len)
        };

        reader.zip(values_slice.iter().copied())
    }

    // Implementazione con enumerate
    #[inline]
    fn get_with_offset_iter_enumerate(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (C, V)> {
        unsafe { assert_unchecked(len > 0) };
        let mut reader = self.compressed_dataset.components.iter_range(offset, len);

        (0..len).map(move |i| {
            let component = reader.next().unwrap();
            let value = unsafe { *self.compressed_dataset.values.get_unchecked(offset + i) };
            (component, value)
        })
    }

    // Implementazione con collect e poi iterate (pre-fetch tutto)
    #[inline]
    fn get_with_offset_iter_collect(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = (C, V)> {
        unsafe { assert_unchecked(len > 0) };
        let reader = self.compressed_dataset.components.iter_range(offset, len);
        let values_slice = unsafe {
            self.compressed_dataset
                .values
                .get_unchecked(offset..offset + len)
        };

        // Collect tutto in un Vec e poi iterate
        reader
            .zip(values_slice.iter().copied())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

fn create_test_data() -> BenchDataset<u16, FixedU16Q> {
    let input_file =
        "/data2/knn_datasets/sparse_datasets/msmarco_v1_passage/cocondenser/data/documents.bin";
    let dataset = SparseDatasetMut::<u16, f32>::read_bin_file(&input_file).unwrap();
    let compressed =
        seismic::compressed_dataset::SparseDatasetCompressed::<u16, FixedU16Q>::from_dataset_f32(
            dataset,
        );

    BenchDataset {
        compressed_dataset: compressed,
    }
}

fn bench_implementations(c: &mut Criterion) {
    let dataset = create_test_data();

    let mut group = c.benchmark_group("get_with_offset_iter");

    // Test con diverse lunghezze
    for len in [10, 50, 100, 500].iter() {
        let offset = 100; // offset fisso per il test

        group.bench_with_input(BenchmarkId::new("generator", len), len, |b, &len| {
            b.iter(|| {
                let iter =
                    dataset.get_with_offset_iter_generator(black_box(offset), black_box(len));
                // Consumiamo l'iteratore
                let sum: f32 = iter.map(|(_, v)| v.to_f32().unwrap()).sum();
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("slice_map", len), len, |b, &len| {
            b.iter(|| {
                let iter =
                    dataset.get_with_offset_iter_slice_map(black_box(offset), black_box(len));
                let sum: f32 = iter.map(|(_, v)| v.to_f32().unwrap()).sum();
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("zip", len), len, |b, &len| {
            b.iter(|| {
                let iter = dataset.get_with_offset_iter_zip(black_box(offset), black_box(len));
                let sum: f32 = iter.map(|(_, v)| v.to_f32().unwrap()).sum();
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("enumerate", len), len, |b, &len| {
            b.iter(|| {
                let iter =
                    dataset.get_with_offset_iter_enumerate(black_box(offset), black_box(len));
                let sum: f32 = iter.map(|(_, v)| v.to_f32().unwrap()).sum();
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("collect", len), len, |b, &len| {
            b.iter(|| {
                let iter = dataset.get_with_offset_iter_collect(black_box(offset), black_box(len));
                let sum: f32 = iter.map(|(_, v)| v.to_f32().unwrap()).sum();
                black_box(sum)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_implementations);
criterion_main!(benches);
