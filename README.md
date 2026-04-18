# FP8 / FP16 GEMM Optimizations

CUDA matrix multiplication comparison across FP32, naive FP16, and WMMA Tensor Core implementations on an NVIDIA A4000 (SM 86).

## Build

Build `libgputk` first, then the project:

```bash
cd libgputk && make && cd ..
make
```

Executables are placed in `build/`.

## Runtime Setup

Set `LD_LIBRARY_PATH` before running any executable (add to `~/.bashrc` to persist):

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/compute/michael.x/FP8-FP16_GEMM_Optimizations/libgputk/lib
```

## Running Executables

All executables take the same argument format:

```
./build/<executable> -i <input0>,<input1> -e <expected> -o <output> -t matrix
```

- `-i`: comma-separated input files (no spaces)
- `-e`: expected output file (for validation)
- `-o`: where to write output
- `-t matrix`: required for `gpuTKSolution` validation

Dataset 1 is small (64×64), dataset 13 is large (~2300×2300).

---

### FP32 Naive GEMM (`basicMatrixMultiplication`)

Baseline FP32 kernel with 16×16 thread blocks.

```bash
./build/basicMatrixMultiplication -i data/1/input0.raw,data/1/input1.raw -e data/1/output.raw -o /tmp/out.raw -t matrix
```

---

### Naive FP16 GEMM (`fp16MatrixMultiplication`)

FP16 multiply-accumulate (`__hmul`/`__hadd`) with FP32 output. No Tensor Cores.

```bash
./build/fp16MatrixMultiplication -i data/1/input0.raw,data/1/input1.raw -e data/1/output.raw -o /tmp/out.raw -t matrix
```

---

### WMMA Tensor Core FP16 GEMM (`fp16WMMAMatrixMultiplication`)

FP16 GEMM using NVIDIA Tensor Cores via the WMMA API. Inputs are padded to multiples of 16.

```bash
./build/fp16WMMAMatrixMultiplication -i data/1/input0.raw,data/1/input1.raw -e data/1/output.raw -o /tmp/out.raw -t matrix
```

---

### Comparison (`compareMatrixMultiplication`)

Runs all three kernels (FP32, naive FP16, WMMA) on the same inputs and prints relative L2 error and max absolute error vs the FP32 reference. `-e` and `-t` are optional for this binary.

```bash
./build/compareMatrixMultiplication -i data/1/input0.raw,data/1/input1.raw -o /tmp/out.raw -t matrix
```

Large dataset:

```bash
./build/compareMatrixMultiplication -i data/13/input0.raw,data/13/input1.raw -o /tmp/out.raw -t matrix
```

## Source Files

| File | Description |
|---|---|
| `src/basicMatrixMultiplication.cu` | FP32 naive GEMM baseline |
| `src/fp16MatrixMultiplication.cu` | Naive FP16 GEMM (`__hmul`/`__hadd`, FP32 output) |
| `src/fp16WMMAMatrixMultiplication.cu` | FP16 WMMA Tensor Core GEMM |
| `src/compareMatrixMultiplication.cu` | Runs all three kernels, prints accuracy vs FP32 |
