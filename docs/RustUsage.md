## Using the Rust Code

This guide explains how to use Seismic's Rust code independently ([standalone](#itself)) or integrate it into your own Rust project ([via Cargo](#notitsef)).

### <a name="itself"> Use Seismic</a>

There are two main types of indices in Rust, each represented by a different data structure.

##### InvertedIndex
This data structure implements the core logic of Seismic. To build an inverted index, you first need to convert the data into Seismic's internal format. This structure ignores document and query IDs, using their indices as identifiers, similar to the well-known `faiss` library.  

You can find usage examples in `src/bin/build_inverted_index` and `src/bin/perf_inverted_index`.

##### SeismicIndex
This is a wrapper around `InvertedIndex`. It internally manages document and query IDs. The index is built by passing it a path to a collection in `jsonl` format, eliminating the need to manually convert it into the internal format.  

Examples of usage can be found in `src/bin/build_enhanced_inverted_index` and `src/bin/perf_enhanced_inverted_index`.

---

## **Detailed Example: InvertedIndex**
The `build_inverted_index` executable constructs an inverted index for a dataset. Both dataset and query files are stored in Seismic's internal binary format.

### **Data Format**

Documents and queries should be JSON-formatted files with the following fields:

- `id`: An integer representing the document ID.
- `vector`: A dictionary where each key represents a token, and its corresponding value is the score, e.g., `{"dog": 2.45}`.

This is the standard output format of several libraries for training sparse models, such as [`learned-sparse-retrieval`](https://github.com/thongnt99/learned-sparse-retrieval).

The script `convert_json_to_inner_format.py` converts files in this format into Seismic's internal format.

```bash
python scripts/convert_json_to_inner_format.py \
    --document-path /path/to/document.jsonl \
    --queries-path /path/to/queries.jsonl \
    --output-dir /path/to/output 
```

This generates a `data` directory at `/path/to/output`, containing `documents.bin` and `queries.bin` binary files. It also saves:
- `doc_ids.npy` and `query_ids.npy`: Containing the original document and query IDs.
- `token_to_id_mapping.json`: Mapping from original tokens (`str`) to token IDs (`u16`).

---

### **Building the Index**
Several parameters control the space/time trade-offs when building the index:

- `--n-postings`: Regulates the posting list size, representing the average number of postings stored per list.
- `--summary-energy`: Controls summary size, preserving a fraction of the overall energy for each summary.
- `--centroid-fraction`: Determines the number of centroids per posting list, capped at a fraction of the posting list length.

For **Splade on MSMarco**, good choices are:
```bash
--n-postings 3500 --summary-energy 0.4 --centroid-fraction 0.1
```

To create a Seismic index serialized in `splade.bin.3500.seismic`, run:

```bash
./target/release/build_inverted_index \
    -i splade.bin \
    -o splade.bin.3500_0.4_0.1 \
    --centroid-fraction 0.1 \
    --summary-energy 0.4 \
    --n-postings 3500
```

---

### **Executing Queries**
Use the `perf_inverted_index` executable to execute queries. Two parameters trade off efficiency and accuracy:

- `--query-cut`: Limits the search algorithm to the top `query_cut` components of the query.
- `--heap-factor`: Skips blocks whose estimated dot product exceeds `heap_factor` times the smallest dot product in the top-k results.

Example command:

```bash
./target/release/perf_inverted_index \
    -i splade.bin.3500_0.4_0.1 \
    -q splade_queries.bin \
    -o results.tsv \
    --query-cut 5 \
    --heap-factor 0.7
```

Queries are executed in **single-thread mode** by default. To enable multithreading, modify the Rust code:

```rust
queries.iter().take(n_queries).enumerate() 
// Change to:
queries.par_iter().take(n_queries).enumerate()
```

The results are written to `results.tsv`. Each query produces `k` lines in the following format:

```text
query_id\tdocument_id\tresult_rank\tdot_product
```

Where:
- `query_id`: A progressive identifier for the query.
- `document_id`: The document ID from the indexed dataset.
- `result_rank`: The ranking of the result by dot product.
- `dot_product`: The similarity score.

---

## **<a name="notitsef">Use Seismic in Your Rust Code</a>**

To incorporate the Seismic library into your Rust project, navigate to your project directory and run:

```bash
cargo add seismic
```

---

## **Creating a Toy Dataset**
Let's create a toy dataset with vectors using `f32` values. We'll then convert it to use half-precision floating points ([`half::f16`](https://docs.rs/half/latest/half/)) and check the dataset's properties.

```rust
use seismic::SparseDataset;
use half::f16;

let data = vec![
    (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    (vec![1, 3],       vec![4.0, 5.0]),
    (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
];

let dataset: SparseDataset<f16> = data.into_iter().collect::<SparseDataset<f32>>().into();

assert_eq!(dataset.len(), 3);  // Number of vectors  
assert_eq!(dataset.dim(), 5);  // Number of components  
assert_eq!(dataset.nnz(), 9);  // Number of non-zero components  
```

To read a dataset in Seismic's internal binary format and quantize values to `f16`:

```rust
let dataset = SparseDataset::<f32>::read_bin_file(&input_filename).unwrap().quantize_f16();
```

---

## **Building and Querying an Index**
Let's build an index using our toy dataset and search for a query.

```rust
use seismic::inverted_index::{Configuration, InvertedIndex};
use seismic::SparseDataset;
use half::f16;

let data = vec![
    (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    (vec![1, 3],       vec![4.0, 5.0]),
    (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
];

let dataset: SparseDataset<f16> = data.into_iter().collect::<SparseDataset<f32>>().into();

let inverted_index = InvertedIndex::build(dataset, Configuration::default());

let result = inverted_index.search(&vec![0, 1], &vec![1.0, 2.0], 1, 5, 0.7);

assert_eq!(result[0].0, 8.0);
assert_eq!(result[0].1, 1);
```

### **Configuration Parameters**
Key parameters to experiment with:
- **`n_postings`** (`PruningStrategy::GlobalThreshold`): Regulates posting list size.
- **`summary_energy`** (`SummarizationStrategy::EnergyPreserving`): Controls summary size.
- **`centroid_fraction`** (`BlockingStrategy::RandomKmeans`): Sets centroids per posting list.

Refer to [Seismic parameters](#parameters) for recommended values.

For serialization/deserialization examples, see:
- [`build_inverted_index.rs`](src/bin/build_inverted_index.rs)
- [`perf_inverted_index.rs`](src/bin/perf_inverted_index.rs)

The `search` method signature:

```rust
pub fn search(
    &self,
    query_components: &[u16],
    query_values: &[f32],
    k: usize,
    query_cut: usize,
    heap_factor: f32,
) -> Vec<(f32, usize)>
```
