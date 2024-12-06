## Package Installation
We provide two options for installing `py-seimic` depending on what is required:

### 1 - Maximum performance
If you want to compile the package optimized for your CPU, you need to install the package from the Source Distribution.
In order to do that you need to have the Rust toolchain installed. Use the following commands:
#### Prerequisites
Install Rust (via `rustup`):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
#### Installation
```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: py-seismic
```
This will compile the Rust code tailored for your machine, providing maximum performance.

### 2 - Easy installation
If you are not interested in obtaining the maximum performance, you can install the package from a prebuilt Wheel.
If a compatible wheel exists for your platform, pip will download and install it directly, avoiding the compilation phase.
If no compatible wheel exists, pip will download the source distribution and attempt to compile it using the Rust compiler (rustc).
```bash
pip install py-seismic
```

Prebuilt wheels are available for Linux platforms (x86_64, i686, aarch64) with different Python implementation (CPython, PyPy) for linux distros using glibc 2.17 or later.
Wheels are also available x86_64 platforms with linux distros using musl 1.2 or later.