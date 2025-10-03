
import pandas as pd
import sys

gt_path = sys.argv[1]
run_path = sys.argv[2]


columns = ["query_id", "doc_id", "rank", "score"]
gt_pd = pd.read_csv(gt_path, sep="\t", names=columns)
run_pd = pd.read_csv(run_path, sep="\t", names=columns)

assert set(gt_pd['query_id']) == set(run_pd['query_id']), "The query ids in the groundtruth and in the results do not match"
assert len(gt_pd) == len(run_pd) == 6980 * 10


def compute_accuracy(gt_pd, res_pd):
    gt_pd_groups = gt_pd.groupby('query_id')['doc_id'].apply(set)
    res_pd_groups = res_pd.groupby('query_id')['doc_id'].apply(set)

    # Compute the intersection size for each query_id in both dataframes
    intersections_size = {
        query_id: len(gt_pd_groups[query_id] & res_pd_groups[query_id]) if query_id in res_pd_groups else 0
        for query_id in gt_pd_groups.index
    }

    # Computes total number of results in the groundtruth
    total_results = len(gt_pd)
    total_intersections = sum(intersections_size.values())
    return total_intersections/total_results

accuracy = compute_accuracy(gt_pd, run_pd)

print(f"Accuracy@10: {accuracy:.4f}")