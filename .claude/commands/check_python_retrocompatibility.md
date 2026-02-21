# Check Python Backward Compatibility

Check that the current branch's Python API is backward compatible with the version currently published on PyPI (`pyseismic-lsr`).

---

## What this check covers and what it does not

**Covered:**
- Removed classes or top-level functions
- Removed methods on existing classes
- New *required* parameters added to existing methods (parameters without a default value)

**Not covered — known limitations:**
- **Return types**: the snapshot captures names and parameter signatures via `__text_signature__`, but PyO3 does not expose return type information at runtime. A method that used to return a 3-tuple and now returns a 2-tuple would not be flagged.
- **Behavioral changes**: if logic changes silently (e.g. a score is computed differently), the snapshot cannot detect it.
- **Type of parameters**: parameter names and positions are compared, but not their annotated types.

**Why not `.pyi` files:**
PyO3 builds "builtin" extension types that standard stub generators (`mypy stubgen`, `griffe`) cannot introspect reliably without manual annotations. `pyo3-stub-gen` exists but requires adding `#[gen_stub_pyclass]` and similar macros throughout the Rust source — a non-trivial change. The `__text_signature__` approach used here works with what PyO3 already exposes and is robust enough to catch the most impactful breaking changes (removals and new required parameters).

---

## Step 1 — Install the published PyPI version and capture its snapshot

```bash
pip install pyseismic-lsr --force-reinstall
```

Then capture the snapshot:

```bash
python -c "
import seismic, re

lines = []
for name in sorted(dir(seismic)):
    if name.startswith('_'):
        continue
    obj = getattr(seismic, name)
    if hasattr(obj, '__mro__'):
        lines.append(name)
        for mname in sorted(dir(obj)):
            if mname.startswith('_'):
                continue
            m = getattr(obj, mname)
            sig = getattr(m, '__text_signature__', '')
            lines.append(f'  .{mname}{sig}')
    else:
        sig = getattr(obj, '__text_signature__', '')
        lines.append(f'{name}{sig}')

print('\n'.join(lines))
" > /tmp/seismic_api_pypi.txt
```

Print the captured snapshot so the user can see the PyPI baseline:

```bash
cat /tmp/seismic_api_pypi.txt
```

---

## Step 2 — Build the current branch and install it

```bash
RUSTFLAGS='-C target-cpu=native' maturin build --release
for whl in target/wheels/*.whl; do python -m pip install "$whl" --force-reinstall; done
```

---

## Step 3 — Capture the current branch's snapshot

```bash
python -c "
import seismic, re

lines = []
for name in sorted(dir(seismic)):
    if name.startswith('_'):
        continue
    obj = getattr(seismic, name)
    if hasattr(obj, '__mro__'):
        lines.append(name)
        for mname in sorted(dir(obj)):
            if mname.startswith('_'):
                continue
            m = getattr(obj, mname)
            sig = getattr(m, '__text_signature__', '')
            lines.append(f'  .{mname}{sig}')
    else:
        sig = getattr(obj, '__text_signature__', '')
        lines.append(f'{name}{sig}')

print('\n'.join(lines))
" > /tmp/seismic_api_current.txt
```

Print the captured snapshot:

```bash
cat /tmp/seismic_api_current.txt
```

---

## Step 4 — Compare and report

Run the following comparison script. It will print a clear pass/fail result.

```bash
python -c "
import re, sys

def parse_required_params(sig_str):
    if not sig_str:
        return set()
    m = re.match(r'\(([^)]*)\)', sig_str)
    if not m:
        return set()
    params = [p.strip().lstrip('\$') for p in m.group(1).split(',')]
    required = set()
    for p in params:
        p = p.strip()
        if not p or p in ('self', '/', '*'):
            continue
        if '=' not in p:
            required.add(p)
    return required

def parse_snapshot(lines):
    api = {}
    current_class = None
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith('  .'):
            rest = line.strip().lstrip('.')
            method_name = re.split(r'[\(\\\$]', rest)[0].strip()
            if current_class is not None:
                api[current_class][method_name] = line.strip()
        else:
            current_class = re.split(r'[\(\\\$]', line)[0].strip()
            api[current_class] = {}
    return api

with open('/tmp/seismic_api_pypi.txt') as f:
    baseline = parse_snapshot(f.read().splitlines())

with open('/tmp/seismic_api_current.txt') as f:
    current = parse_snapshot(f.read().splitlines())

breaking = []

for name, methods in baseline.items():
    if name not in current:
        breaking.append(f'REMOVED class/function: {name}')
        continue
    for method, bline in methods.items():
        if method not in current[name]:
            breaking.append(f'REMOVED method: {name}.{method}')
            continue
        bsig = re.search(r'(\(.*\))', bline)
        csig = re.search(r'(\(.*\))', current[name][method])
        if bsig and csig:
            old_req = parse_required_params(bsig.group(1))
            new_req = parse_required_params(csig.group(1))
            added = new_req - old_req - {'self'}
            if added:
                breaking.append(
                    f'NEW REQUIRED PARAM(S) in {name}.{method}: {sorted(added)}'
                )

if breaking:
    print('BACKWARD COMPATIBILITY BROKEN:')
    for b in breaking:
        print(f'  x {b}')
    sys.exit(1)
else:
    print('OK: current branch is backward compatible with the published PyPI version.')
"
```

---

## Step 5 — Report

After the comparison:

- If the check **passes**: confirm to the user that the API is backward compatible and summarise any *additions* (new methods or optional parameters) as informational.
- If the check **fails**: list every breaking change clearly, group them by class, and explain what code in downstream consumers (e.g. SentenceTransformers) would break and why. Do **not** suggest fixes automatically — report first and wait for instructions.
