## Using the Rust Code
This guide explains how to use Seismic's Rust code independently ([standalone](#itself)) or integrate it into your own Rust project ([via Cargo](#notitsef)).


### <a name="itself"> Using Seismic Binary Executable</a>
First, we clone the Seismic Git repository:

```bash
git clone git@github.com:TusKANNy/seismic.git
cd seismic
```

Then, we have to compile the project. After executing the following command, the binary executable will be found in `./target/release`. 

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

#### Building the Index
Let's now build an index on the Splade embeddings for the MS MARCO v1 passage. To download the encoded vectors, please refer to [Setting up for the Experiment](RunExperiments.md#bin_data).

Seismic has few parameters that control the space/time trade-offs when building the index:

- `--n-postings` defines the posting list size, representing the average number of postings stored per list.
- `--summary-energy` controls the summary size, preserving a fraction of the overall energy for each summary.
- `--centroid-fraction` determines the number of centroids per posting list, capped at a fraction of the posting list length.
- `--clustering-algorithm` selects the algorithm to cluster postings within each posting list.

For **Splade on MS MARCO **, good choices are:
```
--n-postings 4000 --summary-energy 0.4 --centroid-fraction 0.1 --clustering-algorithm random-kmeans-inverted-index-approx
```

To create a Seismic index serialized in the file `documents.bin.4000_0.4_0.1.index.seismic`, run:

```bash
./target/release/build_inverted_index \
    -i ~/sparse_datasets/msmarco_v1_passage/cocondenser/data/documents.bin \
    -o ~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/documents.bin.4000_0.4_0.1 \
    --centroid-fraction 0.1 \
    --summary-energy 0.4 \
    --n-postings 4000 \
    --clustering-algorithm random-kmeans-inverted-index-approx
```

#### Executing Queries
To query the index we need to use the `perf_inverted_index` executable. Two parameters trade off efficiency and accuracy:

- `--query-cut` limits the search algorithm to the top `query_cut` components of the query.
- `--heap-factor` skips blocks whose estimated dot product exceeds `heap_factor` times the smallest dot product in the top-k results.

Example command:

```bash
./target/release/perf_inverted_index \
    -i ~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/documents.bin.4000_0.4_0.1.index.seismic \
    -q ~/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin \
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

## <a name="notitsef">Use Seismic in Your Rust Code</a>

To incorporate the Seismic library into your Rust project, navigate to your project directory and run:

```bash
cargo add seismic
```

### Creating a Toy Dataset
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

### Building and Querying an Index
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

### Configuration Parameters
Key parameters to experiment with:
- `n_postings` (`PruningStrategy::GlobalThreshold`) defines the posting list size.
- `summary_energy` (`SummarizationStrategy::EnergyPreserving`) controls summary size.
- `centroid_fraction` (`BlockingStrategy::RandomKmeans`) sets centroids per posting list.

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
