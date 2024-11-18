import argparse
import json
from tqdm import tqdm
import os
import struct
import numpy as np
import gzip


def write_sparse_vectors_to_binary_file(filename, term_id):
    # A binary sequence is a sequence of integers prefixed by its length,
    # where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.
    # Followed by a sequence of f32, with the same length
    def write_binary_sequence(lst_pairs, file):
        file.write((len(lst_pairs)).to_bytes(4, byteorder="little", signed=False))
        for v in lst_pairs:
            file.write((int(v[0])).to_bytes(4, byteorder="little", signed=False))
        for v in lst_pairs:
            value = v[1]
            ba = bytearray(struct.pack("f", value))
            file.write(ba)

    with open(filename, "wb") as fout:
        fout.write((len(term_id)).to_bytes(4, byteorder="little", signed=False))
        for d in tqdm(term_id):
            lst = sorted(list(d.items()))
            write_binary_sequence(lst, fout)


def write_sparse_vectors_to_binary_file_2(filename, term_id):
    # A binary sequence is a sequence of integers prefixed by its length,
    # where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.
    # Followed by a sequence of f32, with the same length
    def write_binary_sequence(lst_pairs, file):
        file.write((len(lst_pairs)).to_bytes(4, byteorder="little", signed=False))
        for v in lst_pairs:
            file.write((int(v[0])).to_bytes(4, byteorder="little", signed=False))
        for v in lst_pairs:
            value = v[1]
            ba = bytearray(struct.pack("f", value))
            file.write(ba)

    with open(filename, "wb") as fout:
        fout.write((len(term_id)).to_bytes(4, byteorder="little", signed=False))
        for dd in tqdm(term_id):
            d = {x: y for x, y in zip(dd[0], dd[1])}
            lst = sorted(list(d.items()))
            write_binary_sequence(lst, fout)


## This is meant to be used for the NQ dataset in our format.
def convert_documents_with_no_token_conversion(document_folder):
    sorted_files = sorted(
        filter(lambda x: x.endswith(".json"), os.listdir(document_folder)),
        key=lambda x: x.split(".", maxsplit=1),
    )
    documents = []
    doc_ids = []
    for current_file in tqdm(sorted_files):
        with open(os.path.join(document_folder, current_file)) as f:
            l = json.load(f)
        documents.extend(
            dict(zip(element["coordinates"], element["values"]))
            for element in l["vectors"]
        )
        doc_ids.extend([element["id"] for element in l["vectors"]])
    return documents, doc_ids


def convert_queries_with_no_token_conversion(queries_path):
    queries = []
    query_ids = []
    with open(queries_path) as f:
        query_file = json.load(f)
    queries.extend(
        dict(zip(element["coordinates"], element["values"]))
        for element in query_file["vectors"]
    )
    query_ids.extend([element["id"] for element in query_file["vectors"]])

    return queries, query_ids


def convert_documents_from_compressed_file(document_path):
    """
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token,
                and its corresponding value is the score.
    """

    tokens_set = set()

    print("Scanning the documents to build the token to ids mapping")

    with gzip.open(document_path, "r") as f:
        # skip first 512 bytes
        f.seek(512)
        for line in tqdm(f):
            try:
                result = json.loads(line)
            except:
                print("Footer skipped\n")
                break

            tokens_set.update(result["vector"].keys())
    # Tokens are sorted to ensure portability
    sorted_tokens_set = sorted(tokens_set)
    token_to_id_mapping = {v: i for i, v in enumerate(list(sorted_tokens_set))}

    documents = []
    doc_ids = []

    with gzip.open(document_path, "r") as file:
        # skip first 512 bytes
        file.seek(512)
        for line in tqdm(file):
            try:
                line_data = json.loads(line.strip())
            except:
                print("Footer skipped\n")
                break
            vs = np.array([v for v in line_data["vector"].values()], dtype=np.float32)
            ks = np.array([token_to_id_mapping[k] for k in line_data["vector"].keys()])
            # new_ks = np.array([token_to_id_mapping[k] for k in ks])
            documents.append((ks, vs))
            doc_ids.append(line_data["id"])
    return documents, doc_ids, token_to_id_mapping

def convert_documents_from_folder(document_path):
    """
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token,
                and its corresponding value is the score.
    """

    tokens_set = set()

    print("Scanning the documents to build the token to ids mapping")
    for file_name in tqdm(os.listdir(document_path)):
        with open(os.path.join(document_path, file_name), "r") as file:
            for line in file:
                line_data = json.loads(line.strip())
                tokens_set.update(line_data["vector"].keys())

    # Tokens are sorted to ensure portability
    sorted_tokens_set = sorted(tokens_set)
    token_to_id_mapping = {v: i for i, v in enumerate(list(sorted_tokens_set))}

    documents = []
    doc_ids = []

    for file_name in tqdm(os.listdir(document_path)):
        with open(os.path.join(document_path, file_name), "r") as file:
            for line in file:
                line_data = json.loads(line.strip())
                vs = np.array([v for v in line_data["vector"].values()], dtype=np.float32)
                ks = np.array([token_to_id_mapping[k] for k in line_data["vector"].keys()])
                # new_ks = np.array([token_to_id_mapping[k] for k in ks])
                documents.append((ks, vs))
                doc_ids.append(line_data["id"])
    return documents, doc_ids, token_to_id_mapping

def convert_documents_from_file(document_path):
    """
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token,
                and its corresponding value is the score.
    """

    tokens_set = set()

    print("Scanning the documents to build the token to ids mapping")

    with open(document_path, "r") as file:
        for line in tqdm(file):
            line_data = json.loads(line.strip())
            tokens_set.update(line_data["vector"].keys())

    # Tokens are sorted to ensure portability
    sorted_tokens_set = sorted(tokens_set)
    token_to_id_mapping = {v: i for i, v in enumerate(list(sorted_tokens_set))}

    documents = []
    doc_ids = []

    with open(document_path, "r") as file:
        for line in tqdm(file):
            line_data = json.loads(line.strip())
            vs = np.array([v for v in line_data["vector"].values()], dtype=np.float32)
            ks = np.array([token_to_id_mapping[k] for k in line_data["vector"].keys()])
            # new_ks = np.array([token_to_id_mapping[k] for k in ks])
            documents.append((ks, vs))
            doc_ids.append(line_data["id"])
    return documents, doc_ids, token_to_id_mapping


def convert_queries_from_compressed_file(queries_path, token_to_id_mapping=None):
    """
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token,
                and its corresponding value is the score.
    """

    queries = []
    queries_ids = []
    with gzip.open(queries_path, "r") as f:
        f.seek(512)
        json_list = list(f)
        for json_str in tqdm(json_list):
            try:
                result = json.loads(json_str.decode())
            except:
                print("Footer skipped\n")
                break
            new_dict = {token_to_id_mapping[k]: v for k, v in result["vector"].items()}
            # integer_id = int(result['id'])
            queries.append(new_dict)
            queries_ids.append(result["id"])

    print(f"Number of queries: {len(queries)}")

    return queries, queries_ids


def convert_queries_from_file(queries_path, token_to_id_mapping=None):
    """
    Documents and queries must be a jsonl (note the final "l") file.
    Each line is a json file with the following fields:
        - "id": must represent the id of the document as an integer
        - "content": the original content of the document, as a string
        - "vector": a dictionary where each key represents a token,
                and its corresponding value is the score.
    """

    with open(queries_path, "r") as f:
        json_list = list(f)

    queries = []
    queries_ids = []
    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        new_dict = {token_to_id_mapping[k]: v for k, v in result["vector"].items()}
        # integer_id = int(result['id'])
        queries.append(new_dict)
        queries_ids.append(result["id"])

    return queries, queries_ids


def main():
    parser = argparse.ArgumentParser(
        description="Parser for documents and queries conversion to Seismic format."
    )

    # Add arguments
    parser.add_argument("--document-path-or-folder", help="Path to the documents file")
    parser.add_argument("--query-path", help="Path to the queries file")
    parser.add_argument(
        "--output-dir",
        help="Path to the output dir. Will create a 'data' repo inside it. ",
    )
    parser.add_argument(
        "--skip-token-conversion",
        action="store_true",
        default=False,
        help="Whether you want to skip the token to id converison (if you already have done it)",
    )

    parser.add_argument(
        "--large-dataset",
        action="store_true",
        default=False,
        help="Set to true if you are dealing with very large datasets and you don't want to load the entire collection in memory.",
    )

    args = parser.parse_args()
    document_path = args.document_path_or_folder
    query_path = args.query_path
    output_dir = args.output_dir

    data_dir = os.path.join(output_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print(f"Saving into {data_dir}")
    # print("Document Path:", document_path)
    # print("Query Path:", query_path)

    if args.large_dataset:
        # Borrow the implementation from here https://github.com/leeeov4/seismic_big_data/blob/main/scripts/convert_gzip_to_inner_format2.py
        raise NotImplementedError("Logic for large datasets not implement yet!")

    if not args.skip_token_conversion:
        print(f"Reading and converting documents from {document_path}")
        if document_path.endswith("tar.gz"):
            documents, doc_ids, token_id_mapping = (
                convert_documents_from_compressed_file(document_path)
            )
        else:
            if os.path.isdir(document_path):
                 documents, doc_ids, token_id_mapping = convert_documents_from_folder(
                document_path
            )
            else: 
                documents, doc_ids, token_id_mapping = convert_documents_from_file(
                document_path
            )

        print(f"Reading and converting queries from {query_path}")
        
        if query_path.endswith("tar.gz"):
            queries, queries_ids = convert_queries_from_compressed_file(
                query_path, token_id_mapping
            )
        else:
            queries, queries_ids = convert_queries_from_file(
                query_path, token_id_mapping
            )

        print("Saving to Seismic format")
        # Saving documents
        seismic_format_doc_path = os.path.join(data_dir, "documents.bin")
        write_sparse_vectors_to_binary_file_2(seismic_format_doc_path, documents)

        # Saving doc_ids
        np.save(os.path.join(data_dir, "doc_ids.npy"), doc_ids)

        # Saving queries
        seismic_format_query_path = os.path.join(data_dir, "queries.bin")
        write_sparse_vectors_to_binary_file(seismic_format_query_path, queries)

        # Saving query_ids
        np.save(os.path.join(data_dir, "queries_ids.npy"), queries_ids)

        # Saving token to id mapping
        token2id_path = os.path.join(data_dir, "token_to_id_mapping.json")
        with open(token2id_path, "w") as fp:
            json.dump(token_id_mapping, fp)

    else:
        
        print("Reading and converting documents...")
        documents, doc_ids = convert_documents_with_no_token_conversion(document_path)
        print("Reading and converting queries...")
        queries, query_ids = convert_queries_with_no_token_conversion(query_path)
        print("Saving to Seismic format")
        seismic_format_doc_path = os.path.join(data_dir, "documents.bin")
        write_sparse_vectors_to_binary_file(seismic_format_doc_path, documents)
        seismic_format_query_path = os.path.join(data_dir, "queries.bin")
        write_sparse_vectors_to_binary_file(seismic_format_query_path, queries)
        np.save(os.path.join(data_dir, "doc_ids.npy"), doc_ids)
        np.save(os.path.join(data_dir, "queries_ids.npy"), query_ids)


if __name__ == "__main__":
    main()
