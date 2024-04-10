<p align="center">
    <img width="200px" src="imgs/logo.jpg" />
    <h1 align="center">Seismic</h1>
</p>



Seismic is designed for effective and efficient retrieval over *learned sparse embeddings*. Pleasantly, the design uses in a new way two familiar data structures: the inverted
and the forward index.  The approach organizes inverted lists into geometrically-cohesive blocks. Each block is equipped with a sketch, serving as a summary of its vectors. The summaries allow us to skip over many blocks during retrieval and save substantial compute. When a summary indicates that a block must be examined, we use the forward index to retrieve exact embeddings of its documents and compute inner products.

The figure below gives an overview of the overall design.

<p align="center">
  <img src="imgs/index.png" width="90%" alt="The design of Seismic.">
</p>

Experimental results show that single-threaded query processing using Seismic, reaches sub-millisecond per-query latency on various sparse embeddings of the MSMarco dataset
while maintaining high recall. The results indicate that Seismic is one to two orders of magnitude faster than state-of-the-art inverted index-based solutions and further outperforms the winning (graph-based) submissions to the BigANN Challenge by a significant margin.

<h2 align="center">Code is coming soon!</h2>

## <a name="bib">Bibliography</a>
1. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini, *"Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations,"* In SIGIR, 2024. 

### Citation License

The source code in this repository is subject to the following citation license:

By downloading and using this software, you agree to cite the under noted paper in any kind of material you produce where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation license.

```bibtex
@inproceedings{Seismic,
  author = {Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini},
  title = {Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations},
  booktitle = {The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval ({SIGIR})}
  publisher = {ACM},
  year = {2024}
}
```
