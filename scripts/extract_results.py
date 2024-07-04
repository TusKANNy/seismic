import pandas as pd
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Parser for extracting the best configuration from a grid search on a single index.")
    parser.add_argument("--file-path", help="Path to the run file")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.file_path, sep="\t")
    
    chosen_configurations = []
    for recall_cut in np.arange(90, 98, 1):
        chosen_configurations.append(df[df.recall > recall_cut].sort_values("time").head(1))
    df = pd.concat(chosen_configurations)
    print(df)
    print(df.to_markdown(tablefmt="grid", index=False))
    


if __name__ == "__main__":
    main()
