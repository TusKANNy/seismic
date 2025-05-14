# Guidelines to Build your Seismic Index
This guide provides step-by-step instructions for building a Seismic Index on your collection. We explain each hyper-parameter involved in the index construction and search processes, and offer practical suggestions for setting them based on your dataset size and desired trade-offs between speed, accuracy, and memory usage.




### Building Hyper-parameters

- `n_postings` Sets the number of doc ids retained (on average) in each posting lists. Increasing this value improves accuracy but also increases the index size. Resonable values for collections around $10$ millions (MsMarco) are in the range $[3000, 6000]$, while for larger collections like MsMarco-v2, we suggest values in $[60000, 80000]$.

- `centroid_fraction`. Sets the number of centroids per list as fraction of `n_postings`. This affects retrieval time, accuracy, and index size. Higher values yield better accuracy at the cost of increased time and memory. We found that resonable values are $\{0.05, 0.1, 0.2\}$. 


- `summary_energy`. The clustering algorithm groups pruned postings lists into clusters and then a summary vector is kept as a representative of the entire list. As the summary vectors tend to grow quickly in space (see the paper for more details), we keep `summary_energy` of the total $L_1$ norm of the summary vector. This parameters influence the accuracy and the index size, with higher values yielding higher accuray and higher index size. Good values for `summary_energy` are in the range $[0.4, 0.6]$


- `min_cluster_size`.Sets the minimum allowed size of each cluster. Elements from smaller clusters are reassigned to larger one. The default value is $2$. 

- `max_fraction`. The posting list pruning strategy works by keeping on average `n_postings` per list. Some lists may be much longer, hence `max_fraction` 
 sets the maximum allowed posting list length as `max_fraction` $\times$ `n_postings`. We found that using $6$ is a good choice.

 - `knn`. If set, defines the number of neighbors to compute per each document in the $\kappa$-NN graph. __Building the k-NN graph takes much longer than the standalone Seismic index__. If you already have a computed $\kappa$-NN graph, you can pass it with the parameters `knn-path`.


 ##### <span style="color:orange">[The following parameters can only che changed from the Rust interface]</span>


- `clustering_algorithm`. Sets the clustering algorithm used to group documents within each pruned posting list. We have currently three clustering algorithms based on dot-product-based K-Means:
    - `RandomKmeans`. Randomly select a `centroid_fraction` $ \times$ `n_postings` number of  centroids per each posting lists and uses them as centroids. Each element in the list is assigned to the centroid maximizing the dot product with it; the dot product is computed exactly.


    - `RandomKmeansInvertedIndex` and `RandomKmeansInvertedIndexApprox`. Both this algorithms approximate `RandomKmeans` with slightly different strategies, yielding a much faster building time, especially on large colleciton. 

    We suggest to use `RandomKmeansInvertedIndexApprox` with `kmeans_doc_cut`=15, __which is set as default construction method to use when building from the python interface__.


 ### Search Hyper-parameters

- `query_cut`. Specifies the number of posting lists to explore during search. Larger values increase effectiveness while reducing efficiency. We found that setting `query_cut` = $10$ generally yields high effectiveness.

- `heap_factor`. Controls how aggressively the block-skip mechanism prunes the search. Higher values result in more aggressive pruning, improving speed but potentially reducing accuracy. We found the best values are usually between $[0.7, 1]$.

- `n_knn`. The number of neighbors to explore per each document in the results provided by the inverted index. This values is only considered if the `knn` graph was built or passed to the index. 

- `sorted`. Whether to scan the summaries of the first posting list by the most promosing one to the least promising. Setting it to `True` increases efficiency, while `False` increases efficiency. 



### Examples

Let’s walk through a practical example using the MS MARCO Passage dataset, encoded with the SPLADE Co-Condenser Distill model. You can find the dataset [here](https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade).

A highly effective index can be built with the following parameters
```python 
index = SeismicIndex.build(
    json_input_file,
    n_postings=3000,
    centroid_fraction=0.2,
    min_cluster_size=2,
    summary_energy=0.5, 
    max_fraction=6)
```
Search using these parameters:

```python
results = index.batch_search(
    queries_ids=queries_ids,
    query_components=query_components,
    query_values=query_values,
    k=10,
    query_cut=10,
    heap_factor=0.8,
    sorted=False,
)
```


This setup yields:
- Accuracy@10: $99$%

- MRR@10: $0.3827$

For comparison, exact search on this dataset gives an MRR@10 of $0.3828$ — demonstrating that this configuration offers near-exact accuracy with significantly better efficiency.