import sys
import pandas as pd

def read_from_file(filename):
    with open(filename, "r") as f:
        gt = []
        for line in f:
            if line.startswith("\t"):
                continue
            if "\t" not in line:
                continue
            l = line.split("\t")
            query_id = int(l[0])
            doc_id = int(l[1])
            if query_id == len(gt):
                gt.append(set())
            gt[query_id].add(doc_id)
    return gt

column_names = ["query_id", "doc_id", "rank", "score"]
gt_pd = pd.read_csv(sys.argv[1], sep='\t', names=column_names , encoding='latin-1')
res_pd = pd.read_csv(sys.argv[2], sep='\t', names=column_names , encoding='latin-1')

# Group both dataframes by 'query_id' and get unique 'doc_id' sets
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

print(f"\nRecall: {total_intersections/total_results*100:.2f}\n")

gt = read_from_file(sys.argv[1])
r = read_from_file(sys.argv[2])

counter = 0
tot = 0
for s1, s2 in zip(gt,r):
    counter += len(s1 & s2)
    tot += len(s1)

recall = counter/tot*100
print(f"\nRecall: {counter/tot*100:.2f}\n")
print(f"{counter/tot*100:.2f}", file=sys.stderr)


