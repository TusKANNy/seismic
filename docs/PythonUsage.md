# Python Usage

##  Installation

## Package Installation
There are two (+1) options for installing Seismic's Python binding `pyseismic-lsr`. 
The first option is easier but may provide suboptimal performance, while the second one is slightly more complicated but runs faster.

There is also a third option: install the Python ibinding for the latest version of Seismic for the Github repository. This version may be still unpublished but includes latest bugfixes.

### Python - Easy installation
If you are not interested in obtaining maximum performance, you can install the package from a prebuilt Wheel.
If a compatible wheel exists for your platform, `pip` will download and install it directly, avoiding the compilation phase.
If no compatible wheel exists, pip will download the source distribution and attempt to compile it using the Rust compiler (rustc).
```bash
pip install pyseismic-lsr
```

Prebuilt wheels are available for Linux platforms (x86_64, i686, aarch64) with different Python implementations (CPython, PyPy) for Linux distros using glibc 2.17 or later.
Wheels are also available on x86_64 platforms with Linux distros using MUSL 1.2 or later.

### Python - Maximum performance
If you want to compile the package optimized for your CPU, install the package from the Source Distribution.

To do that you need to have the Rust toolchain installed. Use the following commands:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then, you can run the following command to compile Seismic and install its Python interface:

```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: pyseismic-lsr
```

This will compile the Rust code tailored for your machine, providing maximum performance.

### Python - Maximum performance with latest version from Github repository

First, we clone the Seismic Git repository:

```bash
git clone git@github.com:TusKANNy/seismic.git
cd seismic
```

Then, we compile and install Seismic with `maturin`:

```bash
pip install maturin
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

## Usage Example in Python

We can first import the required packages. 

```python
from seismic import SeismicIndex
import numpy as np
import json
import ir_datasets
from ir_measures import *
```

Then, we can load a jsonl file containing the embeddings of our documents and build the Seismic index.

```python
json_input_file = "" # your input file

index = SeismicIndex.build(json_input_file)
print("Number of documents: ", index.len)
print("Avg number of non-zero components: ", index.nnz / index.len)
print("Dimensionality of the vectors: ", index.dim)

index.print_space_usage_byte()
```

We are now ready to read the encoded queries from an input jsonl file.

```python
queries_path = "" # your query file

queries = []
with open(queries_path, 'r') as f:
    for line in f:
        queries.append(json.loads(line))

MAX_TOKEN_LEN = 30
string_type  = f'U{MAX_TOKEN_LEN}'

queries_ids = np.array([q['id'] for q in queries], dtype=string_type)

query_components = []
query_values = []

for query in queries:
    vector = query['vector']
    query_components.append(np.array(list(vector.keys()), dtype=string_type))
    query_values.append(np.array(list(vector.values()), dtype=np.float32))
```

We can now perform our queries in batch.

```python
results = index.batch_search(
    queries_ids=queries_ids,
    query_components=query_components,
    query_values=query_values,
    k=10,
    query_cut=20,
    heap_factor=0.7,
    sorted=True,
    n_knn=0,
)
```

The top-10 documents matching the queries are stored in `results`.
The final step is to use `ir_measures` for the evaluation of the final results.

```python
ir_results = [ir_measures.ScoredDoc(query_id, doc_id, score) for r in results for (query_id, score, doc_id) in r]
qrels = ir_datasets.load('msmarco-passage/dev/small').qrels

ir_measures.calc_aggregate([RR@10], qrels, ir_results)
```
