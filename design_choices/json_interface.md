
### Inverted Index Wrapper
The scope of this .md file is to provide the requirements for a structs that wraps the functionalities of the existing `InvertedIndexBase`. This struct will be referred as wrapper from now on. The name of the struct could be `SeismicIndex` but we may decide that later. A (probably) working implementation of the `SeismicIndex` wrapper exists in `src/inverted_index_wrapper.rs`. You can look at that to better understand the requirements, but I would like to improve it from several perspectives, including code quality, documentation, efficiency (if possible). I will now list the functionalities that I expect from that class below.

- The role of that struct is to wrap the `InvertedIndexBase` struct, in order to make it easier to be used in retrieval settings, hiding under-the-hood efficiency optimizations for those that are not interested in ANN on sparse vectors but just want to quickly use this codebase.
- Seismic (the name of the data structure implemented in this repository) is used to retrieve over learned sparse representations. Here, document and queries are lists (or dictionaries) of `token`: `value` pairs.

    - `token` a string in the tokenizer vocabulary used by the model,  
    - `value` is a floating point value tha represent the saliency of the token.

- Tokens strings are converted into an internal integer-based mapping for efficiency purposes. The wrapper make this operations transparent to the user by reading the data in the `jsonl` format.

- Documents and queries should have the following format. Each line should be a JSON-formatted string with the following fields:
    - `id`: must represent the ID of the document as an integer.
    - `content`: the original content of the document, as a string. This field is optional. 
    - `vector`: a dictionary where each key represents a token, and its corresponding value is the score, e.g., `{"dog": 2.45}`.
- the utilities to read `jsonl` should be improved as well, I feel like the existing ones are suboptimal. 


- the wrapper should keep track of the input document ids (`id` field above) , as it does in the current implementation. This is required to remap them when returning the results, as in the `remap_doc_ids` function. 


Let's plan these intervations together. Before moving on to the implementation, I want to carefully discuss the design choice. 

