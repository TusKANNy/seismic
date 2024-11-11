import argparse
import sys
import os
import subprocess
import re
from datetime import datetime
from termcolor import colored
import json

# from run_experiments import *
from run_experiments import (
    compute_accuracy,
    compile_rust_code,
    compute_metric,
    get_git_info,
    parse_toml,
)


def query_execution(
    configs,
    experiment_dir,
    index_file,
    query_cut,
    heap_factor,
    first_sorted=False,
    knn=0,
):
    """Execute a query based on the provided configuration."""

    query_file = os.path.join(configs["folder"]["data"], configs["filename"]["queries"])
    subsection_name = (
        f"qc_{query_cut}_hp_{heap_factor}_knn_{knn}_first_sorted_{first_sorted}"
    )
    output_file = os.path.join(experiment_dir, f"results_{subsection_name}")
    log_output_file = os.path.join(experiment_dir, f"log_{subsection_name}")

    query_command = configs.get("query-command", "./target/release/perf_inverted_index")

    command_and_params = [
        configs["settings"]["NUMA"] if "NUMA" in configs["settings"] else "",
        query_command,
        f"--index-file {index_file}.index.seismic",
        f"-k {configs['settings']['k']}",
        f"--query-file {query_file}",
        f"--query-cut {query_cut}",
        f"--heap-factor {heap_factor}",
        f"--n-runs {configs['settings']['n-runs']}",
        f"--output-path {output_file}",
        f"--n-knn {knn}",
    ]

    if first_sorted:
        command_and_params.append("--first-sorted")

    command = " ".join(command_and_params)

    print(f"Executing query for subsection '{subsection_name}' with command:")
    print(command)

    pattern = r"\tTotal: (\d+) Bytes"  # Pattern to match the total memory usage

    query_time = 0
    # Run the query and display output in real-time
    print(f"Running query for subsection: {subsection_name}...")
    with open(log_output_file, "w") as log:
        query_process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in iter(query_process.stdout.readline, b""):
            decoded_line = line.decode()
            if decoded_line.startswith("Time ") and decoded_line.strip().endswith(
                "microsecs per query"
            ):
                query_time = int(decoded_line.split()[1])

            match = re.search(pattern, decoded_line)
            if match:
                memory_usage = int(match.group(1))
            print(decoded_line, end="")  # Print each line as it is produced
            log.write(decoded_line)  # Write each line to the output file
        query_process.stdout.close()
        query_process.wait()

    if query_process.returncode != 0:
        print(f"Query execution for subsection '{subsection_name}' failed.")
        sys.exit(1)

    print(f"Query for subsection '{subsection_name}' executed successfully.")

    gt_file = os.path.join(
        configs["folder"]["data"], configs["filename"]["groundtruth"]
    )
    metric = configs["settings"]["metric"]
    return (
        query_time,
        compute_accuracy(output_file, gt_file),
        compute_metric(configs, output_file, gt_file, metric),
        memory_usage,
    )


def get_index_filename_simple(base_filename, config):
    """Generate the index filename based on the provided parameters."""
    name = [
        base_filename,
        "n-postings",
        config["n-postings"],
        "centroid-fraction",
        config["centroid-fraction"],
        "summary-energy",
        config["summary-energy"],
        "knn",
        config["knn"],
    ]

    return "_".join(str(l) for l in name)


def build_and_run(
    input_file,
    index_folder,
    base_filename,
    current_config,
    overall_config,
    base_experiment_dir,
):
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
    index_name = get_index_filename_simple(base_filename, current_config)
    index_file_path = os.path.join(index_folder, index_name)
    experiment_dir = os.path.join(base_experiment_dir, index_name)
    os.makedirs(experiment_dir, exist_ok=False) # Fails if already exists

    print(colored("Build Index", "blue"))
    print(f"Dataset filename: {input_file }")
    print(f"Index filename: {index_file_path}")
    print(f"Saving log in: {experiment_dir}")

    build_command = overall_config.get(
        "build-command", "./target/release/build_inverted_index"
    )

    command_and_params = [
        build_command,
        f"--input-file {input_file}",
        f"--output-file {index_file_path}",
        f"--n-postings {current_config['n-postings']}",
        f"--summary-energy {current_config['summary-energy']}",
        f"--centroid-fraction {current_config['centroid-fraction']}",
        f"--knn {current_config['knn']}",
        f"--clustering-algorithm {current_config['clustering-algorithm']}",
        f"--kmeans-pruning-factor {current_config['kmeans-pruning-factor']}",
        f"--kmeans-doc-cut {current_config['kmeans-doc-cut']}",
    ]

    command = " ".join(command_and_params)

    # Print the command that will be executed
    print("Building index with command:")
    print(command)

    building_output_file = os.path.join(experiment_dir, "building.output")

    # Build the index and display output in real-time
    print("Building index...")
    with open(building_output_file, "w") as build_output:
        build_process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in iter(build_process.stdout.readline, b""):
            decoded_line = line.decode()
            print(decoded_line, end="")  # Print each line as it is produced
            build_output.write(decoded_line)  # Write each line to the output file
        build_process.stdout.close()
        build_process.wait()

    if build_process.returncode != 0:
        print("Index building failed.")
        sys.exit(1)

    print("Index built successfully.\n\n\n")

    print(colored("Searching on index:", "blue"))
    metric = overall_config["settings"]["metric"]
    print(f"Evaluation with metric {metric}")

    # Execute queries for each subsection under [query]
    with open(os.path.join(experiment_dir, "report.tsv"), "w") as report_file:
        report_file.write(
            f"Query Cut\tHeap Factor\tknn\tFirt sorted\tQuery Time (microsecs)\tRecall\t{metric}\tMemory Usage (Bytes)\n"
        )
        for query_cut in overall_config["querying_parameters"]["query-cuts"]:
            for heap_factor in overall_config["querying_parameters"]["heap-factors"]:
                for knn in overall_config["querying_parameters"]["knns"]:
                    for first_sorted in overall_config["querying_parameters"][
                        "first_sorted"
                    ]:
                        print(
                            "Running config: ",
                            query_cut,
                            heap_factor,
                            knn,
                            first_sorted,
                        )
                        query_time, recall, metric, memory_usage = query_execution(
                            configs=overall_config,
                            experiment_dir=experiment_dir,
                            index_file=index_file_path,
                            query_cut=query_cut,
                            heap_factor=heap_factor,
                            knn=knn,
                            first_sorted=first_sorted,
                        )
                        report_file.write(
                            f"{query_cut}\t{heap_factor}\t{knn}\t{first_sorted}\t{query_time}\t{recall}\t{metric}\t{memory_usage}\n"
                        )


def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print("Error: Configuration data is empty.")
        sys.exit(1)

    # Get the experiment name from the configuration
    experiment_name = config_data.get("name")
    print(f"Running experiment: {experiment_name}")

    # Create an experiment folder with date and hour
    timestamp = str(datetime.now()).replace(" ", "_")
    experiment_folder = os.path.join(
        config_data["folder"]["experiment"], f"{experiment_name}_{timestamp}"
    )
    os.makedirs(experiment_folder, exist_ok=True)
    # Store the output of the Rust compilation and index building processes
    get_git_info(experiment_folder)
    compile_rust_code(experiment_folder, config_data)

    print(colored("Grid search information:", "yellow"))
    
    print(json.dumps(config_data["indexing_parameters"], indent=4))
    input_file = os.path.join(
        config_data["folder"]["data"], config_data["filename"]["dataset"]
    )
    index_folder = config_data["folder"]["index"]
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
    print(f"Saving indexes in.. {index_folder}")
    print("\n\n")
    for n_postings in config_data["indexing_parameters"]["n-postings"]:
        for centroid_fraction in config_data["indexing_parameters"][
            "centroid-fraction"
        ]:
            for summary_energy in config_data["indexing_parameters"]["summary-energy"]:
                for knn in config_data["indexing_parameters"]["knn"]:
                    for clust_alg in config_data["indexing_parameters"][
                        "clustering-algorithm"
                    ]:
                        for kmeans_doc_cut in config_data["indexing_parameters"][
                            "kmeans-doc-cut"
                        ]:
                            for kmeans_pruning_factor in config_data[
                                "indexing_parameters"
                            ]["kmeans-pruning-factor"]:
                                current_config = {}
                                current_config["n-postings"] = n_postings
                                current_config["centroid-fraction"] = centroid_fraction
                                current_config["summary-energy"] = summary_energy
                                current_config["knn"] = knn
                                current_config["clustering-algorithm"] = clust_alg
                                current_config["kmeans-doc-cut"] = kmeans_doc_cut
                                current_config["kmeans-pruning-factor"] = (
                                    kmeans_pruning_factor
                                )
                                build_and_run(
                                    input_file,
                                    index_folder,
                                    config_data["filename"]["index"],
                                    current_config,
                                    config_data,
                                    experiment_folder,
                                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid search of seismic experiments on a dataset and find the best configurations to query it."
    )
    parser.add_argument(
        "--exp", required=True, help="Path to the experiment configuration TOML file."
    )
    args = parser.parse_args()

    main(args.exp)
