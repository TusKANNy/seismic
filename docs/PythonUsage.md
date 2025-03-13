## Usage Example in Python
```python
from seismic import SeismicIndex
import numpy as np
import json
import ir_measures
import ir_datasets
from ir_measures import *
```

Build the index

```python
json_input_file = "" # your input file

index = SeismicIndex.build(json_input_file)
print("Number of documents: ", index.len)
print("Avg number of non-zero components: ", index.nnz / index.len)
print("Dimensionality of the vectors: ", index.dim)

index.print_space_usage_byte()
```

Load queries

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

Perform the search

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

Evaluation

```python
ir_results = [ir_measures.ScoredDoc(query_id, doc_id, score) for r in results for (query_id, score, doc_id) in r]
qrels = ir_datasets.load('msmarco-passage/dev/small').qrels

ir_measures.calc_aggregate([RR@10], qrels, ir_results)
```
