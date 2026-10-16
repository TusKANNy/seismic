# Longer token strings in the Python interface

- **Token dtype widened from `U30` to `U100`.** `get_seismic_string()` now
  returns `'U100'` and every token-string argument of the Python classes
  (document tokens, query components, query ids) expects that width. NumPy
  silently truncates strings that exceed the fixed width, so with `U30` two
  distinct terms sharing their first 30 characters (common for word-level
  encoders on corpora with long terms or URLs) collapsed onto the same token
  and `SeismicDataset.add_document` crashed with the opaque vectorium assert
  "Components must be given in strictly ascending order". Colliding terms are
  not merged on purpose: scores must match an exact dot product on the
  original terms. The wider dtype costs 400 bytes per element instead of 120,
  only for the Python-side arrays; the index stores integer ids and is
  unaffected.
  → `MAX_TOKEN_LEN`, `SEISMIC_STRING` in `src/pylib/mod.rs`

  **Breaking for callers that hardcode `'U30'`.** An array with any other
  dtype is rejected with a `TypeError` that names the argument, the dtype
  received and the one expected. Build arrays with
  `dtype=get_seismic_string()`, or convert existing ones with
  `.astype(get_seismic_string())`.

  **Indexes built with 0.5.3 or earlier.** Their token map holds the strings
  truncated to 30 characters. Loaded in 0.5.4, a query token longer than 30
  characters arrives in full, matches no entry and is silently dropped:
  lower recall, no error. Rebuild such indexes with 0.5.4, or truncate query
  tokens to 30 characters until then.

- **`add_document` raises `ValueError` on duplicate tokens** instead of
  tripping the vectorium assert. The message names the document id and the
  token, and adds the fixed-width truncation explanation when a token of the
  document fills the whole width. The check runs before anything is stored.
  Previously the rejected document's id was recorded without its vector, so a
  caller that caught the exception and kept adding documents got every later
  result labelled with the wrong document id. The Rust
  `SeismicDataset::add_document` now returns `Result<(), String>`.
  → `SeismicDataset::add_document` in `src/inverted_index_wrapper.rs`

- **No Rust panic on a wrong array dtype.** `batch_search` (index classes
  and datasets) extracted each per-query array with `unwrap()`, so a wrong
  dtype inside the list surfaced as a `PanicException`. It is now a
  `TypeError` that names the offending list entry.
  → `extract_tokens`, `extract_values` in `src/pylib/mod.rs`

# Faster index construction

Optimizations to the index build. None of them changes the built index: the
produced posting lists, blocks, and summaries are identical to the previous
implementation, up to the arbitrary choice among postings tied at the global
pruning threshold.

- **Parallel global-threshold pruning.** The global threshold is found by
  sorting all posting scores in parallel instead of scanning them through a
  serial heap; all postings scoring at or above the threshold are then
  distributed to components in parallel, and the surplus of scores tied at
  the threshold is dropped deterministically (thread-count-independently)
  from the tails of the posting lists. The old serial implementation is kept
  as a test oracle.
  → `global_threshold_pruning` in `src/inverted_index.rs`

- **Longest-first (LPT) posting-list dispatch.** Posting lists are built
  longest-first so a long list picked up late cannot leave the other workers
  idle; results are restored to component order afterwards.
  → the parallel loop in `InvertedIndexBase::build`, `src/inverted_index.rs`

- **Dense scratch for summaries.** The per-block `HashMap` used to accumulate
  component maxima is replaced by a dense epoch-stamped scratch table reused
  across all blocks of a posting list, and top-k selection uses
  `select_nth_unstable_by` instead of heap-based `k_largest_by`. Above 2^20
  components — where allocating the dense table for every posting list would
  dominate the build — the scratch falls back to an `FxHashMap`-backed slot
  table with the same layout, producing identical summaries.
  → `SummaryScratch`, `accumulate_block_max`, `fixed_size_summary`,
  `energy_preserving_summary` in `src/posting_list.rs`

- **CSR inverted index for the clustering.** The temporary centroid index used
  by the random k-means is now a flat offsets + postings layout (one contiguous
  array instead of one allocation per component), built with a counting sort,
  with scores converted to `f32` once at build time.
  → `FastInvertedIndex` in `src/utils.rs`

- **Survivors-only reassignment pass.** After dissolving too-small clusters,
  documents are reassigned against an index holding only the surviving
  centroids instead of the full one, shrinking the score accumulator, the
  posting traversal, and the argmax by the same factor. Assignments are
  bit-for-bit identical to the full-index version.
  → second pass of `do_random_kmeans_on_docids_ii_approx_dot_product` in
  `src/utils.rs`

- **Exact-dot-product k-means on the same index.** The exact variant of the
  clustering builds its pruned centroid index with `FastInvertedIndex` as well
  (score-based per-component pruning via `FastInvertedIndex::pruned`) instead
  of per-component vectors.
  → `do_random_kmeans_on_docids_ii_dot_product` in `src/utils.rs`
