<p align="center">
    <img width="200px" src="imgs/logo.jpg" />
    <h1 align="center">Seismic</h1>
</p>

<p align="center">
    <a href="-"><img src="https://badgen.net/static/paper/SIGIR 2024/green" /></a>
    <a href="https://arxiv.org/"><img src="https://badgen.net/static/arXiv/00984389948/red" /></a>
    <a href="https://https://zenodo.org/"><img src="https://badgen.net/static/Datasets/zenodo/gray" /></a>

</p>

<p align="center">
<a href="https://crates.io/crates/qwt"><img src="https://badgen.infra.medigy.com/crates/v/qwt" /></a>
<a href="https://crates.io/crates/qwt"><img src="https://badgen.infra.medigy.com/crates/d/qwt" /></a>
<a href="LICENSE.md"><img src="https://badgen.net/static/license/MIT/blue" /></a>
</p>


Seismic is designed for effective and efficient retrieval over *learned sparse embeddings*. Pleasantly, the design uses in a new way two familiar data structures: the inverted
and the forward index.  The approach organizes inverted lists into geometrically-cohesive blocks. Each block is equipped with a sketch, serving as a summary of its vectors. The summaries allow us to skip over many blocks during retrieval and save substantial compute. When a summary indicates that a block must be examined, we use the forward index to retrieve exact embeddings of its documents and compute inner products.

The figure below gives an overview of the overall design.

<p align="center">
  <img src="imgs/index.png" width="80%" alt="The design of Seismic.">
</p>

Experimental results show that single-threaded query processing using Seismic, reaches sub-millisecond per-query latency on various sparse embeddings of the MSMarco dataset
while maintaining high recall. The results indicate that Seismic is one to two orders of magnitude faster than state-of-the-art inverted index-based solutions and further outperforms the winning (graph-based) submissions to the BigANN Challenge by a significant margin.

<h2 align="center">Code is coming soon!</h2>

```bibtex
@inproceedings{Seismic,
  author = {Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini},
  title = {Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations},
  booktitle = {The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval ({SIGIR})}
  publisher = {ACM},
  year = {2024}
}
```
