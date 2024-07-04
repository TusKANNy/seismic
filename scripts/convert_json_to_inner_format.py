
import argparse
import json 
from tqdm import tqdm
import os
import struct 
import numpy as np

def write_sparse_vectors_to_binary_file(filename, term_id):
    # A binary sequence is a sequence of integers prefixed by its length, 
    # where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.
    # Followed by a sequence of f32, with the same length
    def write_binary_sequence(lst_pairs, file): 
        file.write((len(lst_pairs)).to_bytes(4, byteorder='little', signed=False))   
        for v in lst_pairs:
            file.write((int(v[0])).to_bytes(4, byteorder='little', signed=False))
        for v in lst_pairs:
            value = v[1]
            ba = bytearray(struct.pack("f", value))  
            file.write(ba) 
    with open(filename, "wb") as fout:
        fout.write((len(term_id)).to_bytes(4, byteorder='little', signed=False))
        for d in tqdm(term_id):
            lst = sorted(list(d.items()))
            write_binary_sequence(lst, fout)


def write_sparse_vectors_to_binary_file_2(filename, term_id):
    # A binary sequence is a sequence of integers prefixed by its length, 
    # where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.
    # Followed by a sequence of f32, with the same length
    def write_binary_sequence(lst_pairs, file): 
        file.write((len(lst_pairs)).to_bytes(4, byteorder='little', signed=False))   
        for v in lst_pairs:
            file.write((int(v[0])).to_bytes(4, byteorder='little', signed=False))
        for v in lst_pairs:
            value = v[1]
            ba = bytearray(struct.pack("f", value))  
            file.write(ba) 
    with open(filename, "wb") as fout:
        fout.write((len(term_id)).to_bytes(4, byteorder='little', signed=False))
        for dd in tqdm(term_id):
            d = {x : y for x, y in zip(dd[0], dd[1])}
            lst = sorted(list(d.items()))
            write_binary_sequence(lst, fout)

## This is meant to be used for the NQ dataset in our format. 
def convert_documents_from_nq(document_folder):
    sorted_files = sorted(filter(lambda x: x.endswith(".json"), os.listdir(document_folder)), key=lambda x: x.split(".", maxsplit=1))
    documents = []
    doc_ids = []
    for current_file in tqdm(sorted_files):
        with open(os.path.join(document_folder, current_file)) as f:
            l = json.load(f)
        documents.extend(
            dict(zip(element['coordinates'], element['values']))
            for element in l['vectors']
        )
        doc_ids.extend([element['id'] for element in l['vectors']])
    return documents, doc_ids

def convert_queries_from_nq(queries_path):
    queries = []
    query_ids = []
    with open(queries_path) as f:
        query_file = json.load(f)
    queries.extend(
        dict(zip(element['coordinates'], element['values']))
        for element in query_file['vectors']
    )
    query_ids.extend([element['id'] for element in query_file['vectors']])
    
    
    return queries, query_ids

def convert_documents_from_splade(document_path):
    '''
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token, 
                and its corresponding value is the score.
    '''

    print("Converting from splade format..")
    with open(document_path, "r") as f:
        json_list = list(f)

    tokens_set = set()

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        tokens_set.update(result['vector'].keys())
    # Tokens are sorted to ensure reproducibility
    sorted_tokens_set = sorted(tokens_set)

    token_to_id_mapping = {v:i for i, v in enumerate(list(sorted_tokens_set))}

    documents = [None] * len(json_list)

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        integer_id = int(result['id'])

        new_dict = { token_to_id_mapping[k]:v for k,v in result['vector'].items() }
        documents[integer_id] = new_dict

        vs = np.array(list(result['vector'].values()), dtype=np.float32)
        ks = np.array(list(result['vector'].keys()))
        new_ks = np.array([token_to_id_mapping[k] for k in ks])

        documents[integer_id] = (new_ks, vs)


    return documents, token_to_id_mapping
    
def convert_queries_from_splade(queries_path, token_to_id_mapping=None):
    '''
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token, 
                and its corresponding value is the score.
    '''

    print("Converting from splade format..")
    with open(queries_path, "r") as f:
        json_list = list(f)
   
    queries = []
    
    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        new_dict = { token_to_id_mapping[k]:v for k,v in result['vector'].items() }
        #integer_id = int(result['id'])
        queries.append(new_dict)

    return queries

def main():
    parser = argparse.ArgumentParser(description="Parser for documents and queries conversion to Seismic format.")

    # Add arguments
    parser.add_argument("--document-path", help="Path to the documents file")
    parser.add_argument("--query-path", help="Path to the queries file")
    parser.add_argument("--output-dir", help="Path to the output dir. Will create a 'data' repo inside it. ")
    parser.add_argument("--input-format", help="Kind of input format, can be msmarco (default) or nq", type=str, choices=["msmarco", "nq"], default="msmarco")
    
    args = parser.parse_args()
    document_path = args.document_path
    query_path = args.query_path
    output_dir = args.output_dir
    
    data_dir = os.path.join(output_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)   
    print(f"Saving into {data_dir}")
    print("Document Path:", document_path)
    print("Query Path:", query_path)

    if args.input_format == "msmarco":
        print("Reading and converting documents...")
        documents, token_id_mapping = convert_documents_from_splade(document_path)
        
        print("Reading and converting queries...")
        queries = convert_queries_from_splade(query_path, token_id_mapping)
        
        print("Saving to Seismic format")
        seismic_format_doc_path = os.path.join(data_dir, "documents.bin")
        write_sparse_vectors_to_binary_file_2(seismic_format_doc_path, documents)
        seismic_format_query_path = os.path.join(data_dir, "queries.bin")
        write_sparse_vectors_to_binary_file(seismic_format_query_path, queries)
        
    elif args.input_format == "nq":
        
        print("Reading and converting documents (NQ format)...")
        documents, doc_ids = convert_documents_from_nq(document_path)
        
        print("Reading and converting queries (NQ format)...")
        queries, query_ids = convert_queries_from_nq(query_path)
        
        print("Saving to Seismic format")
        seismic_format_doc_path = os.path.join(data_dir, "documents.bin")
        write_sparse_vectors_to_binary_file(seismic_format_doc_path, documents)
        seismic_format_query_path = os.path.join(data_dir, "queries.bin")
        write_sparse_vectors_to_binary_file(seismic_format_query_path, queries)
        np.save(os.path.join(data_dir, "doc_ids.npy"), doc_ids)
        np.save(os.path.join(data_dir, "queries_ids.npy"), query_ids)
        

    

   

if __name__ == "__main__":
    main()



