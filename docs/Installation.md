## Package Installation
We provide several options for installing `pyseismic-lsr` and `seismic` depending on what is required:

### Python - Maximum performance
If you want to compile the package optimized for your CPU, you need to install the package from the Source Distribution.
In order to do that you need to have the Rust toolchain installed. Use the following commands:
#### Prerequisites
Install Rust (via `rustup`):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
#### Installation
```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: py-seismic
```
This will compile the Rust code tailored for your machine, providing maximum performance.

### Python - Easy installation
If you are not interested in obtaining the maximum performance, you can install the package from a prebuilt Wheel.
If a compatible wheel exists for your platform, `pip` will download and install it directly, avoiding the compilation phase.
If no compatible wheel exists, pip will download the source distribution and attempt to compile it using the Rust compiler (rustc).
```bash
pip install pyseismic-lsr
```

Prebuilt wheels are available for Linux platforms (x86_64, i686, aarch64) with different Python implementation (CPython, PyPy) for linux distros using glibc 2.17 or later.
Wheels are also available x86_64 platforms with linux distros using musl 1.2 or later.

### Rust 

This command allows you to compile all the Rust binaries contained in `src/bin`

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

There are two main kind of indexes in Rust, represented by two different data structures.

##### InvertedIndex.
This data structure contains the logic of Seismic. To build an inverted index, you first need to convert the data into the inner seismic format. This data structures ignores document and query ids, using their indexes as identifier (as the well-known `faiss` library does). You can find examples on how to use it in `src/bin/build_inverted_index` and `src/bin/perf_inverted_index`.


##### SeismicIndex
This is a wrapper around `InvertedIndex`. It handles document ids and query ids internally. The index is built by passing it the path to collection as `jsonl`, without the need to convert it into the inner format. You can find examples on how to use it in `src/bin/build_enhanced_inverted_index` and `src/bin/perf_enhanced_inverted_index`

This command produces three executables: `build_inverted_index`, `perf_inverted_index`, and `generate_groundtruth` in the `/target/release/` directory.



The `build_inverted_index` executable is used to construct an inverted index for a dataset. Both dataset and query files are stored in an internal binary format. Refer to the Python scripts section for a script to convert a dataset from JSON format. This process involves several parameters that regulate space/time trade-offs:

- `--n-postings`: Regulates the size of the posting list, representing the average number of postings stored per posting list.
- `--summary-energy`: Controls the size of the summaries, preserving a fraction of the overall energy for each summary.
- `--centroid-fraction`: Determines the number of centroids built for each posting list, capped at a fraction of the posting list length.

For Splade on MSMarco, good choices are `--n-postings 3500`, `--summary-energy 0.4`, and `--centroid-fraction 0.1`.

The following command can be used to create a Seismic index serialized in the file `splade.bin.3500.seismic`:

```bash
./target/release/build_inverted_index -i splade.bin -o splade.bin.3500_0.4_0.1 --centroid-fraction 0.1 --summary-energy 0.4 --n-postings 3500 
```

To execute a set of queries, use the `perf_inverted_index` executable. Two parameters, `query-cut` and `heap-factor`, trade-off efficiency vs accuracy:

- `--query-cut`: The search algorithm considers only the top `query_cut` components of the query.
- `--heap-factor`: The search algorithm skips a block whose estimated dot product is greater than `heap_factor` times the smallest dot product of the top-k results in the current heap.

The following command exemplifies this:

```bash
./target/release/perf_inverted_index -i splade.bin.3500_0.4_0.1c -q splade_queries.bin -o results.tsv --query-cut 5 --heap-factor 0.7
```

The dataset of queries is in binary internal format. Refer again to the Python scripts section for a script to convert a dataset from JSON format.

The executable prints the average running time per query. Queries are executed in single-thread mode. To enable multithreading, modify the Rust code by replacing the iteration on the query from `queries.iter().take(n_queries).enumerate()` to `queries.par_iter().take(n_queries).enumerate()`.

The results are written in the file `results.tsv`. For each query, there are `k` lines, one for each of its results. Each line follows this format:

```text
query_id\tdocument_id\tresult_rank\tdot_product
```

Here, `query_id` is a progressive identifier for the query, `document_id` is the identifier of the document in the indexed dataset, and `result_rank` indicates their rank in the ordering by their `dot_product` with the query.