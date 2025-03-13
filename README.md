<h1 align="center">Seismic</h1>
<p align="center">
    <img width="400px" src="https://raw.githubusercontent.com/TusKANNy/seismic/main/imgs/new_logo_seismic.webp" />
</p>

<p align="center">
    <a href="https://dl.acm.org/doi/pdf/10.1145/3626772.3657769"><img src="https://badgen.net/static/paper/SIGIR 2024/green" /></a>  
    <a href="https://dl.acm.org/doi/pdf/10.1145/3627673.3679977"><img src="https://badgen.net/static/paper/CIKM 2024/blue" /></a>
    <a href="https://arxiv.org/abs/2501.11628"><img src="https://badgen.net/static/paper/ECIR 2025/yellow" /></a>
    <a href="http://arxiv.org/abs/2404.18812"><img src="https://badgen.net/static/arXiv/2404.18812/red" /></a>
</p>

<p align="center">    
    <a href="https://crates.io/crates/seismic"><img src="https://badgen.infra.medigy.com/crates/v/seismic" /></a>
    <a href="https://crates.io/crates/seismic"><img src="https://badgen.infra.medigy.com/crates/d/seismic" /></a>
    <a href="LICENSE.md"><img src="https://badgen.net/static/license/MIT/blue" /></a>
</p>

Seismic is a highly efficient data structure for fast retrieval over *learned sparse embeddings* written in Rust 🦀. Designed with scalability and performance in mind, Seismic makes querying learned sparse representations seamless.

Details on how to use Seismic's core engine in Rust 🦀 can be found in [`docs/RustUsage.md`](docs/RustUsage.md).

The instructions below explain how to use it by using the Python API. 


### ⚡ Installation  
To install Seismic, run:

```bash
pip install pyseismic-lsr
```

Check out the detailed installation guide in [docs/Installation.md](docs/Installation.md) for performance optimizations.


### 🚀 Quick Start  
Given a collection as a `jsonl` file, you can quickly index it by running 
```python
from seismic import SeismicIndex

json_input_file = "" # Your data collection

index = SeismicIndex.build(json_input_file)
print("Number of documents: ", index.len)
print("Avg number of non-zero components: ", index.nnz / index.len)
print("Dimensionality of the vectors: ", index.dim)

index.print_space_usage_byte()
```

and then exploit Seismic to retrieve your set of queries quickly

```python
import numpy as np

MAX_TOKEN_LEN = 30

string_type  = f'U{MAX_TOKEN_LEN}'

query = {"a": 3.5, "certain": 3.5, "query": 0.4}
query_id = "0"
query_components = np.array(list(query.keys()), dtype=string_type)
query_values = np.array(list(query.values()), dtype=np.float32)

results = index.search(
    query_id=query_id,
    query_components=query_components,
    query_values=query_values,
    k=10, 
    query_cut=3, 
    heap_factor=0.8,
)
```


### 📥 Download the Datasets  
The embeddings in ```jsonl```  format for several encoders and several datasets can be downloaded from this HuggingFace [repository](https://huggingface.co/collections/tuskanny/seismic-datasets-6610108d39c0f2299f20fc9b), together with the queries representations. 

As an example, the Splade embeddings for MSMARCO can be downloaded and extracted by running the following commands.

```bash
wget https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade/resolve/main/documents.tar.gz?download=true -O documents.tar.gz 

tar -xvzf documents.tar.gz
```

or by using the Huggingface dataset download [tool](https://huggingface.co/docs/hub/en/datasets-downloading).


### 📄 Data Format  
Documents and queries should have the following format. Each line should be a JSON-formatted string with the following fields:
- `id`: must represent the ID of the document as an integer.
- `content`: the original content of the document, as a string. This field is optional. 
- `vector`: a dictionary where each key represents a token, and its corresponding value is the score, e.g., `{"dog": 2.45}`.

This is the standard output format of several libraries to train sparse models, such as [`learned-sparse-retrieval`](https://github.com/thongnt99/learned-sparse-retrieval).

The script ```convert_json_to_inner_format.py``` allows converting files formatted accordingly into the ```seismic``` inner format.

```bash
python scripts/convert_json_to_inner_format.py --document-path /path/to/document.jsonl --queries-path /path/to/queries.jsonl --output-dir /path/to/output 
```
This will generate a ```data``` directory at the ```/path/to/output``` path, with ```documents.bin``` and ```queries.bin``` binary files inside.

If you download the NQ dataset from the HuggingFace repo, you need to specify ```--input-format nq``` as it uses a slightly different format. 


### Resources
Check out our `docs` folder for more detailed guide on use to use Seismic directly in Rust, replicate the results of our paper, or use Seismic with your custom collection. 


### <a name="bib">📚 Bibliography</a>
1. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. "*Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations*." Proc. ACM SIGIR. 2024. 
2. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. "*Pairing Clustered Inverted Indexes with κ-NN Graphs for Fast Approximate Retrieval over Learned Sparse Representations*."  Proc. ACM CIKM. 2024.
3. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini, and Leonardo Venuta. "*Investigating the Scalability of Approximate Sparse Retrieval Algorithms to Massive Datasets*." Proc. ECIR. 2025. *To Appear*. 


### Citation License
The source code in this repository is subject to the following citation license:

By downloading and using this software, you agree to cite the under-noted paper in any kind of material you produce where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation license.

SIGIR 2024
```bibtex
@inproceedings{bruch2024seismic,
  author    = {Bruch, Sebastian and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
  title     = {Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations},
  booktitle = {Proceedings of the 47th International {ACM} {SIGIR} {C}onference on Research and Development in Information Retrieval ({SIGIR})},
  pages     = {152--162},
  publisher = {{ACM}},
  year      = {2024},
  url       = {https://doi.org/10.1145/3626772.3657769},
  doi       = {10.1145/3626772.3657769}
}
```

CIKM 2024
```bibtex 
@inproceedings{bruch2024pairing,
  author    = {Bruch, Sebastian and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano and Venuta, Leonardo},
  title     = {Pairing Clustered Inverted Indexes with $\kappa$-NN Graphs for Fast Approximate Retrieval over Learned Sparse Representations},
  booktitle = {Proceedings of the 33rd International {ACM} {C}onference on {I}nformation and {K}nowledge {M}anagement ({CIKM})},
  pages     = {3642--3646},
  publisher = {{ACM}},
  year      = {2024},
  url       = {https://doi.org/10.1145/3627673.3679977},
  doi       = {10.1145/3627673.3679977}
}
```