import argparse
import sys
import os

import json
from datetime import datetime
from termcolor import colored

import itertools

# from run_experiments import 
from run_experiments import (
    run_experiment,
    parse_toml,
)

def generate_indexing_parameters_combinations(params):
    # Extract keys and values from the dictionary
    keys, values = zip(*params.items())
    # Generate all possible combinations
    all_combinations = itertools.product(*values)
    
    unique_combinations = set()
    for combination in all_combinations:
        combo_dict = dict(zip(keys, combination))
        
        # If 'clustering-algorithm' is 'random-kmeans', set 'kmeans-doc-cut' and 'kmeans-pruning-factor' to a fake value. 
        # These parameters are not needed.
        if combo_dict["clustering-algorithm"] == "random-kmeans":
            combo_dict["kmeans-doc-cut"] = 0
            combo_dict["kmeans-pruning-factor"] = 0.0

        # If 'clustering-algorithm' is 'random-kmeans-inverted-index-approx', set 'kmeans-pruning-factor' to a fake value. 
        # This parameter is not needed.
        if combo_dict["clustering-algorithm"] == "random-kmeans-inverted-index-approx":
            combo_dict["kmeans-pruning-factor"] = 0.0
        
        # Convert dict to tuple of sorted items for deduplication and add to set
        combo_tuple = tuple(sorted(combo_dict.items()))
        unique_combinations.add(combo_tuple)
    
    # Convert back to list of dictionaries
    return [dict(combo) for combo in unique_combinations]

def generate_query_combinations(params):
    keys, values = zip(*params.items())
    all_combinations = itertools.product(*values)
    
    combination_dict = {}
    for i, combination in enumerate(all_combinations, start=1):
        combo_key = f"combination_{i}"
        combo_dict = dict(zip(keys, combination))
        combination_dict[combo_key] = combo_dict
    
    return combination_dict


def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print("Error: Configuration data is empty.")
        sys.exit(1)

    # Get the experiment name from the configuration
    grid_name = config_data.get("name")
    print(f"Running Grid: {grid_name}")

    # Create an experiment folder with date and hour
    timestamp = str(datetime.now()).replace(" ", "_")
    grid_folder = os.path.join(
        config_data["folder"]["experiment"], f"{grid_name}_{timestamp}"
    )
    os.makedirs(grid_folder, exist_ok=True)
    
    print()
    print(colored("Grid search information:", "yellow"))
    
    print(json.dumps(config_data["indexing_parameters"], indent=4))

    query_combinations = generate_query_combinations(config_data["querying_parameters"])

    print("Run an experiment for each building configuration")

    for i, building_config in enumerate(generate_indexing_parameters_combinations(config_data["indexing_parameters"])):
        print()
        print(f"Running buiding combination {i} with {config_data['indexing_parameters']}")
        print(f"Running buiding combination {i} with {json.dumps(config_data['indexing_parameters'], indent=4)}")

        experiment_config = {}
        experiment_config["folder"] = config_data["folder"]
        experiment_config["folder"]["experiment"] = grid_folder

        experiment_config["filename"] = config_data["filename"]
        experiment_config["settings"] = config_data["settings"]

        experiment_config["name"] = f"building_combination_{i}"

        experiment_config["query"] = query_combinations

        experiment_config["indexing_parameters"] = building_config
        with open(os.path.join(grid_folder, f"building_combination_{i}.json"), "w") as f:
            json.dump(building_config, f, indent=4)
        run_experiment(experiment_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid search of seismic experiments on a dataset and find the best configurations to query it."
    )
    parser.add_argument(
        "--exp", required=True, help="Path to the grid configuration TOML file."
    )
    args = parser.parse_args()

    main(args.exp)
