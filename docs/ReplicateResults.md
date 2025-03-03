## Replicate Results

We provide a quick way to replicate the results of our papers. 

As a first step, follow the instructions in `docs/RustUsage` to convert the data into the Seismic inner format. 

Then, use the python script `scripts/run_experiments.py` to quickly reproduce a result from a given paper. Here is an example

```bash
python script/run_experiments.py --exp experiments/sigir2024/splade.toml 
```
Make sure to provide the correct path `folder.data` and `folder.qrels_path` field in the `toml` file. With this command, you can replicate the results on the splade cocondenser embeddings reported in SIGIR2024 paper. 
