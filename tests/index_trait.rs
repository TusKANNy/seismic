//! Seismic through vectorium's generic `Index` trait: alone, and as the first stage of a
//! `RerankIndex`, the way kANNolo and TACHIOM indexes are used.

use seismic::inverted_index::Configuration;
use seismic::{InvertedIndexBase, SearchParams, SeismicIndex};
use vectorium::core::rerank_index::{RerankIndex, RerankSearchParams};
use vectorium::{
    DatasetGrowable, DotProduct, FlatIndex, Index, IndexStats, PlainSparseDataset,
    PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView,
};

const DIM: usize = 300;
const N_DOCS: usize = 3000;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Sorted, deduplicated random sparse vector with 5..=24 nonzeros.
fn random_vector(state: &mut u64) -> (Vec<u16>, Vec<f32>) {
    let nnz = 5 + (lcg(state) % 20) as usize;
    let mut components: Vec<u16> = (0..nnz).map(|_| (lcg(state) % DIM as u64) as u16).collect();
    components.sort_unstable();
    components.dedup();
    let values = components
        .iter()
        .map(|_| (lcg(state) % 1000) as f32 / 1000.0 + 0.01)
        .collect();
    (components, values)
}

fn random_dataset(seed: u64) -> PlainSparseDataset<u16, f32, DotProduct> {
    let mut state = seed;
    let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(DIM, DIM);
    let mut dataset = PlainSparseDatasetGrowable::new(quantizer);
    for _ in 0..N_DOCS {
        let (c, v) = random_vector(&mut state);
        dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
    }
    dataset.into()
}

/// Written against the trait only, as kANNolo/vectorium-generic code would be.
fn search_generic<I: Index>(
    index: &I,
    query: I::Query<'_>,
    k: usize,
    params: &I::SearchParams,
) -> Vec<u64> {
    index
        .search(query, k, params)
        .iter()
        .map(|r| r.vector)
        .collect()
}

#[test]
fn index_trait_matches_inherent_search() {
    let index = InvertedIndexBase::build(random_dataset(7), Configuration::default());
    assert_eq!(IndexStats::n_elements(&index), N_DOCS);
    assert_eq!(IndexStats::dim(&index), DIM);

    let params = SearchParams::new(10, 0.7);
    let mut state = 42;
    for _ in 0..20 {
        let (c, v) = random_vector(&mut state);
        let query = SparseVectorView::new(c.as_slice(), v.as_slice());
        let via_trait = search_generic(&index, query, 10, &params);
        let inherent: Vec<u64> = index
            .search(query, 10, params.query_cut, params.heap_factor, 0, false)
            .iter()
            .map(|r| r.vector)
            .collect();
        assert_eq!(via_trait, inherent);
    }
}

#[test]
fn seismic_index_wrapper_implements_index() {
    let index = SeismicIndex::new(
        random_dataset(7),
        Configuration::default(),
        None::<Vec<String>>,
        None::<Vec<Option<String>>>,
        Default::default(),
    );
    assert_eq!(IndexStats::n_elements(&index), N_DOCS);

    let mut state = 42;
    let (c, v) = random_vector(&mut state);
    let query = SparseVectorView::new(c.as_slice(), v.as_slice());
    let results = search_generic(&index, query, 10, &SearchParams::default());
    assert_eq!(results.len(), 10);
}

/// Seismic retrieves candidates, an exact copy of the collection reranks them; the result
/// should be close to brute force.
#[test]
fn seismic_as_rerank_first_stage() {
    let first_stage = InvertedIndexBase::build(random_dataset(7), Configuration::default());
    let rerank = RerankIndex::new(first_stage, random_dataset(7));
    let flat = FlatIndex::from(random_dataset(7));

    let params = RerankSearchParams::new(100, SearchParams::new(20, 0.5), ());
    let k = 10;
    let n_queries = 50;
    let mut hits = 0;
    let mut state = 42;
    for _ in 0..n_queries {
        let (c, v) = random_vector(&mut state);
        let query = SparseVectorView::new(c.as_slice(), v.as_slice());

        let got = Index::search(&rerank, (query, query), k, &params);
        let exact = flat.search(query, k, &());
        assert_eq!(got.len(), k);
        let exact_ids: Vec<u64> = exact.iter().map(|r| r.vector).collect();
        hits += got.iter().filter(|r| exact_ids.contains(&r.vector)).count();
    }
    let recall = hits as f64 / (k * n_queries) as f64;
    eprintln!("seismic -> rerank recall@{k}: {recall:.3}");
    assert!(recall >= 0.9, "recall@{k} = {recall:.3}");
}
