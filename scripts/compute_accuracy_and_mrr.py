
import argparse
import ir_measures
from ir_measures import *
import pandas as pd
import numpy as np
import sys

def compute_mrr(filename, qrels_path, queries_path):
    with open(filename) as f:
        res = f.readlines()
    res_to_write = [r for r in res if len(r.split("\t")) > 2]
    new_file_name = f"{filename}_temp.tsv"
    with open(new_file_name, "w") as f:
        for x in res_to_write:
            f.write(x)
    df_run = pd.read_csv(
                new_file_name, sep='\t', 
                names=['query_id', 'doc_id', 'rank', 'score'], 
                engine='c'
    )

    df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "useless", "doc_id", "relevance"])
    if queries_path.endswith(".tsv"):    
        queries_small = pd.read_csv(queries_path, sep="\t", names=["query_id", "query"] )
        query_ids = queries_small.query_id.values
    if queries_path.endswith(".npy"):
        query_ids = np.load(queries_path).astype(int)

    
    mapping_to_queries_id = {i: v for i,v in enumerate(query_ids) }
    df_run['query_id'] = df_run['query_id'].apply(lambda x: mapping_to_queries_id[x])
    mrr = ir_measures.calc_aggregate([RR@10], df_qrels, df_run)[RR@10]
    #print("MRR@10: ", mrr)
    return mrr
            
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

def read_stats_from_file(filename):
    r_line = ""
    with open(filename, "r") as f:
        stats = {}
        for line in f:
            if line.startswith("["):
                r_line = line
            if line.startswith("stats"):
                l = line.split(" ")
                for i in range(1,len(l), 2):
                    if l[i] not in stats:
                        stats[l[i]] = []
                    try:
                        stats[l[i]].append(int(l[i+1]))
                    except:
                        print(line, l[i+1])
            else: 
                if "\t" not in line:
                    print(line.strip())
    return stats, r_line

def main():
    parser = argparse.ArgumentParser(description="Parser for computing recall and mrr@10 on MSMARCO.")
    parser.add_argument("--run-path", help="Path to the run file")
    parser.add_argument("--gt-path", help="Path to the groundtruth file.")
    parser.add_argument("--qrels-path", help="Path to the qrels file.")
    parser.add_argument("--original-queries-path", help="Path to the original queries file.")
    
    args = parser.parse_args()
    run_path = args.run_path
    gt_path  = args.gt_path
    
    result = read_from_file(run_path)
    gt = read_from_file(gt_path)
    
    
    qrels_path = args.qrels_path
    queries_path = args.original_queries_path # For qids re-mapping

    counter = 0
    tot = 0
    for s1, s2 in zip(gt, result):
        counter += len(s1 & s2)
        tot += len(s1)

    recall = counter/tot*100
    #print(f"\nRecall: {counter/tot*100:.2f}\n")
    recall = counter/tot*100
    mrr = compute_mrr(run_path, qrels_path, queries_path)
    print("\n")
    print("Recall@10: {:.2f}\nmRR@10: {:.2f}".format(recall,mrr),)
    print("\n")
    
    print(f"{recall}\t{mrr}", file=sys.stderr)
    


if __name__ == "__main__":
    main()