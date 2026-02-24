# 🏆 Best Results: Optimized Configurations Guide


For each dataset and memory budget, we provide **optimal TOML configuration files** that represent the **fastest configuration capable of reaching a specific recall target** while respecting the given memory constraints.

These configurations are the result of extensive grid search experiments to find the  optimal trade off between:
- **Query Speed**: Minimizing retrieval latency
- **Recall**: Achieving target effectiveness levels  
- **Memory Usage**: Respecting memory constraints as multiples of the forward index size

**What is the memory budget?**: We measure memory usage in terms of memory budgets. The memory budgets (1.5×, 2.0×) refer to multiples of the forward index size. As an example, for Splade-Cocondenser on MS-MARCO, whose forward index weighs 4GB, this means:
- `mem_budget_1.5` ≈ 6 GB total memory usage
- `mem_budget_2` ≈ 8 GB total memory usage

## 🗂️ Configuration Structure

```bash
experiments/best_configs/
├── msmarco-v1/                            # Dataset
│   ├── splade-cocondenser/                # Sparse encoder model
│   │   ├── mem_budget_1.5/                # 1.5× forward index memory budget (~6GB)
│   │   │   ├── recall_90.toml             # Fastest config for ≥90% recall
│   │   │   ├── recall_91.toml             # Fastest config for ≥91% recall
│   │   │   ├── ...
│   │   │   └── recall_99.toml             # Fastest config for ≥99% recall
│   │   └── mem_budget_2/                  # 2.0× forward index memory budget (~8GB)
│   │       ├── recall_90.toml
│   │       ├── ...
│   │       └── recall_99.toml
│   └── e-splade/
│       └── efficient-splade/              # Efficient-SPLADE encoder
│           ├── mem_budget_1.5/
│           └── mem_budget_2/
└── msmarco-v2/                            # MS MARCO v2 dataset
    └── cocondenser/
        ├── mem_budget_1.5/
        └── mem_budget_2.0/
```

### Configuration Naming Convention

Each file follows the pattern:
- **Dataset**: `msmarco-v1`, `msmarco-v2`, etc.
- **Model**: `splade-cocondenser`, `efficient-splade`, etc.  
- **Memory Budget**: `mem_budget_1.5` (1.5× forward index), `mem_budget_2` (2.0× forward index)
- **Recall Target**: `recall_90` (≥90%), `recall_95` (≥95%), etc.

## 🚀 Quick Start

### 1. Choose Your Configuration

Select the configuration that matches your requirements:

```bash
# For balanced performance with 1.5× memory budget (~6GB)
cp experiments/best_configs/msmarco-v1/splade-cocondenser/mem_budget_1.5/recall_95.toml my_experiment.toml

# For higher performance with 2.0× memory budget (~8GB)
cp experiments/best_configs/msmarco-v1/splade-cocondenser/mem_budget_2/recall_95.toml my_experiment.toml
```

### 2. Configure Your Paths

Edit the copied TOML file to specify your data paths:

```toml
[folder]
data = "/path/to/your/data/directory"           # Directory containing documents.bin, queries.bin
index = "/path/to/your/index/directory"         # Directory where index will be stored  
qrels_path = "/path/to/your/qrels.dev.tsv"      # Ground truth relevance file
experiment = "/path/to/output/directory"        # Directory for experiment results
```

### 3. Run the Experiment

Use the `run_experiments` script to execute your configuration:

```bash
# Build and run the experiment
python3 scripts/run_experiments.py --exp my_experiment.toml
```
This will create a directory in `folder.experiment` containing the details of the execution, including a `report.tsv` file with results in terms of effectiveness, efficiency, and memory consumpution. 
