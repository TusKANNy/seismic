# Release Checklist

You are helping prepare a release of the Seismic library.
Work through each step below in order. For each step, report what you found and what (if anything) needs to be fixed before proceeding.

---

## Step 0 — Git alignment check

This command is always run from the `main` branch of `seismic_private`. Before doing anything else, verify that `main` and `develop` are fully aligned.

Run the following commands:

```bash
git fetch origin
git log origin/main..origin/develop --oneline   # commits in develop not yet in main
git log origin/develop..origin/main --oneline   # commits in main not yet in develop
```

Both commands must produce **empty output**. If either has commits, stop and report:
- Which branch is ahead and by how many commits.
- The list of diverging commits.
- Do **not** proceed with the rest of the checklist until the branches are aligned.

Also confirm the current branch is `main`:
```bash
git branch --show-current
```

---

## Step 1 — Identify what changed

Run `git log main..HEAD --oneline` and `git diff main -- src/` to get a clear picture of all Rust/Python source changes since the last release on `main`.
Summarise the changes in a short bullet list so you know what to look for in the documentation checks below.

---

## Step 2 — Documentation alignment check

The documentation lives in `docs/`. Cross-check every doc file against the current source code (`src/`, `Cargo.toml`, `pyproject.toml`).
For each file, read both the doc and the relevant source, then report every discrepancy you find.

### 2a — `docs/PythonUsage.md`

Check against `src/pylib/mod.rs` (the `#[pymethods]` blocks). Verify:

- **Class names** exported to Python: `SeismicIndex`, `SeismicIndexLV`, `SeismicIndexRaw`, `SeismicIndexRawLV`, `SeismicDataset`, `SeismicDatasetLV`. Any new or renamed class must be reflected in the doc.
- **`SeismicIndex.build()`** — parameter list, names, types, and default values must match the `#[pyo3(signature = ...)]` attribute exactly.
- **`SeismicIndex.build_from_dataset()`** — same checks.
- **`SeismicIndex.search()`** — parameters and return type. The current return type is `list[tuple[str, float, str]]` i.e. `(query_id, distance, document_id)` with **no** inline content field. If the doc still mentions a 4th `content` element, flag it.
- **`SeismicIndex.batch_search()`** — parameters (including `num_threads`) and return type.
- **`SeismicIndex.get_doc_text(doc_id)`** — this method exists and is the correct way to retrieve document text; verify it is documented.
- **`SeismicIndex.save()` / `.load()`** — paths and extensions (`.index.seismic`).
- **`SeismicIndex.build_knn()` / `.save_knn()` / `.load_knn()`** — signatures.
- **`SeismicIndexRaw`** — if documented, verify its `build()`, `search()`, and `batch_search()` signatures. Note that `batch_search` for `SeismicIndexRaw` takes a `query_path: str` (binary file), not lists of arrays.
- **`get_seismic_string()`** / **`get_string_type()`** helper — check whether the doc uses the correct function name.
- Any example code that references old parameter names (e.g. `sorted=True` added later, `num_threads` added later) or old return tuple shapes.

### 2b — `docs/RustUsage.md`

Check against `src/inverted_index.rs`, `src/configurations.rs`, and the binary entry points in `src/bin/`.

- **`InvertedIndex::build(dataset, Configuration::default())`** — is the import path still `seismic::inverted_index::{Configuration, InvertedIndex}`?
- **`InvertedIndex::search()`** — verify the signature shown in the doc matches the actual Rust method.
- **`SparseDataset`** constructor / `read_bin_file` / `quantize_f16` — confirm these are still the correct API entry points.
- Binary executable names (`build_inverted_index`, `perf_inverted_index`) — confirm they still exist in `src/bin/`.
- CLI flags (`--n-postings`, `--summary-energy`, `--centroid-fraction`, `--clustering-algorithm`, `--query-cut`, `--heap-factor`) — cross-check with the actual binary source.

### 2c — `docs/Guidelines.md`

- Parameter names use `snake_case` in the Python interface (`n_postings`, `centroid_fraction`, etc.) and `kebab-case` on the CLI (`--n-postings`). Make sure the doc is consistent.
- Clustering algorithm names — cross-check the string literals used in the doc against the `ClusteringAlgorithm` enum variants in `src/configurations.rs`.
- Default values mentioned in the doc — cross-check against `#[pyo3(signature = ...)]` defaults in `src/pylib/mod.rs`.

### 2d — `docs/TomlInstructions.md`

- TOML field names — cross-check against how `scripts/run_experiments.py` reads them.
- `value-type` options (`f16`, `bf16`, `f32`, `fixedu8`, `fixedu16`) — confirm these are still the accepted values.
- `component-type` options (`u16`, `u32`) — confirm.
- `clustering-algorithm` string values — cross-check against `src/configurations.rs`.
- Any script names referenced (e.g. `scripts/run_experiments.py`, `scripts/run_grid_search.py`, `scripts/gather_grid_search_results.py`, `scripts/find_best_grid_results_by_recall_range.py`) — verify these files exist with `ls scripts/`.

### 2e — `docs/BestResults.md`

- Folder structure diagram — verify it matches the actual layout under `experiments/best_configs/`.
- Script references — same as above.

### 2f — `README.md`

- Skim the README for any method signatures, parameter names, or feature descriptions that have become stale.
- Check version numbers or package names (`pyseismic-lsr`) for anything that needs updating.

---

## Step 3 — Examples alignment check

The examples live in `examples/`. These notebooks are the first thing a new user runs, so correctness is critical.
For each notebook, read its cells and cross-check API calls against `src/pylib/mod.rs`.

### Notebooks to check

- `examples/HandsOnSeismic.ipynb` — gentle intro, targets new users
- `examples/SeismicGuide.ipynb` — detailed how-to, covers all features
- `examples/RAG.ipynb` — RAG pipeline example using `get_doc_text()`
- `examples/LargeVocabulary.ipynb` — `SeismicIndexLV` / `SeismicDatasetLV` 
- `examples/DotVByteIndex.ipynb` - compressed index usage
usage

> `examples/ReadListExperiments.ipynb` is an internal research notebook; skip it.

### Checks to apply to each notebook

**Imports**
- All imported names (`SeismicIndex`, `SeismicIndexLV`, `SeismicDataset`, `SeismicDatasetLV`, `SeismicIndexRaw`, `SeismicIndexRawLV`) must match the classes actually exported by the module (see `src/lib.rs` `#[pymodule]` block).
- `seismic.get_seismic_string()` is the correct helper function name — flag any use of the old `get_string_type()` alias.

**`build()` and `build_from_dataset()` calls**
- Parameter names and default values must match the current `#[pyo3(signature = ...)]` exactly.
- Flag any parameter that has been added or removed since the notebook was last updated (e.g. `batched_indexing`, `load_content`, `num_threads`).

**`search()` calls**
- The `return_content` parameter **no longer exists** — any call using it must be flagged as broken.
- Current signature: `search(query_id, query_components, query_values, k, query_cut, heap_factor, n_knn=0, sorted=True)`.
- Return type is `list[tuple[str, float, str]]` — three elements `(query_id, score, doc_id)`. Flag any cell output or unpacking that shows or expects a 4th `content` element.

**`batch_search()` calls**
- For `SeismicIndex` / `SeismicIndexLV`: takes `queries_ids, query_components, query_values, k, query_cut, heap_factor, n_knn=0, sorted=True, num_threads=0`.
- For `SeismicIndexRaw` / `SeismicIndexRawLV`: takes `query_path, k, query_cut, heap_factor, n_knn, sorted, num_threads=0` (a file path, not arrays).
- Return type for both is `list[list[tuple[str, float, str]]]` — three-element tuples.

**`get_doc_text()` usage**
- Any example that displays or uses document text must call `index.get_doc_text(doc_id)`, not rely on a 4th element from `search()` / `batch_search()` results.

**Evaluation boilerplate**
- Result unpacking in `ir_measures` loops must match the 3-tuple shape: `for (query_id, score, doc_id) in r`.

**Hard-coded absolute paths**
- Flag any cell containing absolute paths to private or machine-specific directories (e.g. `/data1/…`, `/data2/cosimorulli/…`). These should be replaced with empty strings or clear placeholder comments before a public release.

**Saved cell outputs**
- If a saved output cell shows a 4-tuple result structure (old API), it is misleading even if the code is correct. Flag it so the output can be cleared or regenerated.

---

## Step 4 — Image audit (`imgs/`)

Run `ls imgs/` to get the full list of image files, then search for each filename across all `.md` and `.ipynb` files (`grep -r <filename> --include="*.md" --include="*.ipynb"`).

For each image file found:
- **Referenced**: note where it is used (relative path or absolute URL — both count).
- **Unreferenced**: flag it. Unreferenced images are dead weight and may be leftover from old experiments or internal slides that should not be in a public release.

Report the full list with a ✓ / ✗ status for each file.

---

## Step 5 — Scripts directory audit (`scripts/`)

### 5a — Jupyter notebooks

Run `find scripts/ -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*"` to list all notebooks that are not gitignored checkpoints.

The **only two notebooks allowed** in `scripts/` are:
- `scripts/Notebook.ipynb`
- `scripts/ExtractResutsGridSearch.ipynb`

Any other `.ipynb` file must be flagged immediately and **not released**. These are likely leftover notebooks from private experiments and must be removed or moved before the public push.

### 5b — PDF files

Run `find scripts/ -name "*.pdf"` to check for PDF files.

There must be **zero PDF files** in `scripts/`. PDFs are typically results of private experiments or internal reports and must never appear in the public repo. Flag every one found.

### 5c — Root-level shell scripts and `justfile`

Run `git ls-files "*.sh" justfile Justfile` to list only the files **tracked by git** (i.e. what would land in the public release).

**Shell scripts (`.sh`)**

For each tracked `.sh` file, read its first 10 lines and classify it:

- **Private experiment launcher** — any script that calls `run_grid_search.py`, `run_experiments.py`, iterates over TOML configs, or references internal dataset paths. These must **not** be released. Flag them.
- **Legitimate public utility** — e.g. a CI helper or a reproducible build script that an end-user would actually need. These are OK, but still list them for confirmation.

**`justfile` / `Justfile`**

A `justfile` is a private development convenience tool and must **not** appear in the public release. If tracked by git, flag it.

For every flagged file, the required action before the public push is one of: delete it, move it to a gitignored location, or add it to `.gitignore`.

---

## Step 6 — Report and fix

After completing all checks above:

1. List every discrepancy found, grouped by file/folder.
2. For each discrepancy, propose the exact correction.
3. Ask me to confirm before making any edits to documentation or notebook files.

Do **not** edit any source code (`.rs` files) during this step — only documentation and notebooks.
