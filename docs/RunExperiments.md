## Run an Experiment with Seismic

The easiest way to run an experiment with Seismic is to use the Python script [`scripts/run_experiments.py`](scripts/run_experiments.py).  
This script is configurable via TOML files, which specify the parameters to build the index and execute queries on it.  
The script measures average query time (in microseconds), recall with respect to the true closest vectors of the query (accuracy@k), MRR with respect to judged qrels, and index space usage (bytes).
These results are available in the file `report.tsv` in a folder created by the script togheter with additional information (details below).

The TOML files used to replicate experiments from our published papers can be found in the [`experiments`](experiments/) folder.  
You can easily write your TOML file by following the instructions [here](docs/TomlInstructions.md).

As an example, let's now run the experiments using the TOML file [`experiments/sigir2024/splade.toml`](experiments/sigir2024/splade.toml), which replicates the results of Seismic with Splade embeddings on the MS MARCO dataset.

### <a name="bin_data">Setting up for the Experiment</a>
Let's start by creating a working directory for the data and indexes.

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
python scripts/run_experiments.py --exp experiments/sigir2024/splade.toml
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
