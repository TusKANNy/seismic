## Run an Experiment with Seismic

The easiest way to run an experiment with Seismic is to use the Python script [`scripts/run_experiments.py`](/scripts/run_experiments.py).
This script is configurable via TOML files, which specify the parameters to build the index and execute queries on it.
The script measures average query time (in microseconds), recall with respect to the true closest vectors of the query (accuracy@k), MRR with respect to judged qrels, and index space usage (bytes).
These results are available in the file `report.tsv` in a folder created by the script togheter with additional information (details below).

Pre-optimized TOML configurations for several datasets can be found in [`experiments/best_configs`](/experiments/best_configs/).
You can easily write your own TOML file by following the instructions [here](/docs/TomlInstructions.md).

As an example, let's now run an experiment using the TOML file [`experiments/best_configs/msmarco-v1/splade-cocondenser/mem_budget_1.5/recall_95.toml`](/experiments/best_configs/msmarco-v1/splade-cocondenser/mem_budget_1.5/recall_95.toml), which runs Seismic with Splade embeddings on the MS MARCO dataset targeting 95% recall.

### Download the Datasets

The embeddings in `jsonl` format for several encoders and several datasets can be downloaded from this HuggingFace [repository](https://huggingface.co/collections/tuskanny/seismic-datasets-6610108d39c0f2299f20fc9b), together with the query representations.

As an example, the Splade embeddings for MSMARCO can be downloaded and extracted by running the following commands.

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

The script `scripts/convert_json_to_inner_format.py` allows converting files formatted accordingly into the `seismic` inner format.

```bash
python scripts/convert_json_to_inner_format.py --document-path /path/to/document.jsonl --query-path /path/to/queries.jsonl --output-dir /path/to/output
```
This will generate a `data` directory at the `/path/to/output` path, with `documents.bin` and `queries.bin` binary files inside.

If you download the NQ dataset from the HuggingFace repo, you need to specify `--input-format nq` as it uses a slightly different format.

### <a name="bin_data">Setting up for the Experiment</a>

The experiment scripts work with Seismic's binary format. Let's start by creating a working directory for the data and indexes.

```bash
mkdir -p ~/sparse_datasets/msmarco_v1_passage/cocondenser/
mkdir -p ~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes
```

We need to download datasets, queries, qrels, etc. as follows. Here, we are downloading Splade embeddings for MS MARCO v1 passage.  
Other datasets are available [here](https://huggingface.co/collections/tuskanny/seismic-datasets-6610108d39c0f2299f20fc9b).  
Note: this requires downloading more than 5 GiB of data.

```bash
cd ~/sparse_datasets/msmarco_v1_passage/cocondenser/
wget https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade-bin/resolve/main/msmarco_v1_passage_cocondenser_v2.tar.gz?download=true -O msmarco_v1_passage_cocondenser_v2.tar.gz
mv ./data/qrels.dev.small.tsv ../ # move qrels one folder up, since they're shared with other encoders
```

Let's uncompress the file using the following command:

```bash
tar -xvzf msmarco_v1_passage_cocondenser_v2.tar.gz
```

### Running the Experiment
We are now ready to run the experiment.

First, clone the Seismic Git repository and compile Seismic:

```bash
cd ~
git clone git@github.com:TusKANNy/seismic.git
cd seismic
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

If needed, install Rust on your machine with the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Now we can run the experiment with the following command:

```bash
python scripts/run_experiments.py --exp experiments/best_configs/msmarco-v1/splade-cocondenser/mem_budget_1.5/recall_95.toml
```

Please install the required Python's libraries with the following command:
```bash
pip install -r scripts/requirements.txt
```

The script will build an index using the parameters in the `[indexing_parameters]` section of the TOML file.  
The index is saved in the directory `~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes`.  
You can change directory names by modifying the `[folders]` section in the TOML file.

Next, the script will query the same index with different parameters, as specified in the `[query]` section.  
These parameters provide different trade-offs between query time and accuracy. In our TOML file, we report the expected accuracy level.

**Important**: if your machine is NUMA, you need to uncomment the NUMA setting in the TOML file and configure it according to your hardware for better performance.

### Getting the Results
The script creates a folder named `splade_cocondenser_msmarco_XXX`, where `XXX` encodes the datetime at which the script was executed. This ensures that each run creates a unique directory.

Inside the folder, you can find the data collected during the experiment.

The most important file is `report.tsv`, which reports *query time*, *RR@10*, *memory usage*, and *build time*.

In the folder you can find data gathered during the experiment.

The most important file is `report.tsv`, which reports *query time*, *recall (accuracy@k)* *RR@10*, *memory usage*, and *building time*.
