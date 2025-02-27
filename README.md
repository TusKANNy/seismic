<h1 align="center">Seismic</h1>
<p align="center">
    <img width="200px" src="imgs/new_logo_seismic.webp" />
    
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

Seismic is an data structure designed for effective and efficient retrieval over *learned sparse embeddings*.


## Installation 

Seismic can be install by running
```bash
pip install py-seismic 

```
Detailed instructions on how to install Seismic to maximize its performance can be found in `docs/Installation.md`


## Quick Start 

Given a collection as a `jsonl` file  (details [here](#data-format)), you can quickly index with
```python
json_input_file = "/data4/lvenuta/splade/data/docs_anserini.jsonl"

index = SeismicIndex.build(json_input_file)
print("Number of documents: ", index.len)
print("Avg number of non-zero components: ", index.nnz / index.len)
print("Dimensionality of the vectors: ", index.dim)

index.print_space_usage_byte()
```

and then exploit Seismic to quickly retrieve your set of queries

```python
MAX_TOKEN_LEN = 30
string_type  = f'U{MAX_TOKEN_LEN}'

query = {"a": 3.5, "certain": 3.5, "query": 0.4}
queries_ids = np.array([0])
query_components = np.array(list(query.keys()), dtype=string_type)
query_values = np.array(list(query.values()), dtype=np.float32))

results = index.batch_search(
    queries_ids=queries_ids,
    query_components=query_components,
    query_values=query_values,
    k=10
)
```







### Download the Datasets

The embeddings in ```jsonl```  format used in our experiments can be downloaded from this HugginFace [repository](https://huggingface.co/collections/tuskanny/seismic-datasets-6610108d39c0f2299f20fc9b), together with the queries representations. 

As an example, the <span style="font-variant:small-caps;">Splade</span> embeddings for <span style="font-variant:small-caps;">MsMarco</span> can be downloaded and extracted by runnning the following commands.

```bash
wget https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade/resolve/main/documents.tar.gz?download=true -O documents.tar.gz 

tar -xvzf documents.tar.gz
```

or by using the Huggingface dataset download [tool](https://huggingface.co/docs/hub/en/datasets-downloading).

### Data Format


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




## <a name="bib">Bibliography</a>
1. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini. "*Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations*." In ACM SIGIR. 2024. 

## Citation License

The source code in this repository is subject to the following citation license:

By downloading and using this software, you agree to cite the under-noted paper in any kind of material you produce where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation license.

```bibtex
@inproceedings{Seismic,
  author = {Sebastian Bruch and Franco Maria Nardini and Cosimo Rulli and Rossano Venturini},
  title = {Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations},
  booktitle = {The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval ({SIGIR})},
  publisher = {ACM},
  year = {2024}
}
```
