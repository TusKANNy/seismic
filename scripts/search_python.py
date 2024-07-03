from seismic import PySeismicIndex
from time import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parser for searching with a pre-built Seismic Index.")
    parser.add_argument("--index-path", help="Path to the index file")
    parser.add_argument("--queries-path", help="Path to the queries in the inner Seismic format")
    
    
    args = parser.parse_args()
    
    
    

    index_path = args.index_path

    print(f"Loading the index from {index_path}..")
    index = PySeismicIndex.load(index_path)
    print("Index loaded")
    query_path = args.queries_path
    

    k = 10

    query_cut = 6

    heap_factor = 0.7
    num_threads = 1

    start = time()
    results = index.batch_search(query_path, k, query_cut, heap_factor, num_threads)
    end = time()

    n_queries = 6980
    print("Number of queries ", n_queries)

    print( ((end - start) / 6980) * 10**6)

#print((results[0] ) )


if __name__ == "__main__":
    main()

