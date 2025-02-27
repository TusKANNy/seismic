## <a name="code">Using the Rust Code</a>

To incorporate the Seismic library into your Rust project, navigate to your project directory and run the following Cargo command:

```bash
cargo add seismic
```

This command adds the Seismic library to your project.

#### Creating a Toy Dataset

Let's create a toy dataset comprising vectors with `f32` values. Next, we'll convert this dataset to use half-precision floating points ([`half::f16`](https://docs.rs/half/latest/half/)). Finally, we'll check the number of vectors, the dimensionality, and the number of non-zero components of the dataset.

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
assert_eq!(dataset.nnz(), 9);  // Number of non zero components
```

The following code shows how to read a dataset in the internal binary format with `f32` values and quantize those values to `f16`. 

```rust,ignore
let dataset = SparseDataset::<f32>::read_bin_file(&input_filename).unwrap().quantize_f16();
```

#### Building and Querying an Index

Let's build an index using the above toy dataset and search for a query.

```rust
use seismic::inverted_index::{Configuration,InvertedIndex};
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

There are building configuration parameters to experiment with. Take a look at [build_inverted_index.rs](src/bin/build_inverted_index.rs) code for an example. 

The most important ones are

- `n_postings` in `PruningStrategy::GlobalThreshold`: Regulates the size of the posting list, representing the average number of postings stored per posting list.
- `summary_energy` in `SummarizationStrategy::EnergyPerserving`: Controls the size of the summaries, preserving a fraction of the overall energy for each summary.
- `centroid_fraction` in `BlockingStrategy::RandomKmeans`: Determines the number of centroids built for each posting list, capped at a fraction of the posting list length.

Refer to [Seismic parameters](#parameters) for recommended values for different datasets.

Take a look at [build_inverted_index.rs](src/bin/build_inverted_index.rs) and [perf_inverted_index.rs](src/bin/perf_inverted_index.rs) for examples to serialize/deserialize an index on a file.  

The signature of the `search` method is 

```rust,ignore
pub fn search(
        &self,
        query_components: &[u16],
        query_values: &[f32],
        k: usize,
        query_cut: usize,
        heap_factor: f32,
    ) -> Vec<(f32, usize)>
```

It accepts a sparse vector for the query (`query_components` and `query_values`), `k` for top-`k` results, and parameters `query_cut` and `heap_factor` for trade-off between accuracy and query time.

- `query_cut`: The search algorithm considers only the top `query_cut` components of the query.
- `heap_factor`: The search algorithm skips a block whose estimated dot product is greater than `heap_factor` times the smallest dot product of the top-k results in the current heap.

Refer to [Seismic parameters](#parameters) for their influence on recall and query time on the different datasets. 
