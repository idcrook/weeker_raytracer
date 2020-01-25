# Install CUDA on Ubuntu 19.10

Graphics card: RTX 2070 Super
 - 40 Turing SM
 - 2560 CUDA units

Tried to follow the docs. didn't like gcc-9; requires gcc-8. However there is a working installer for CUDA 10.1 and toolkit available on Ubuntu 19.10


```
# already # sudo apt install nvidia-driver-435
sudo apt install nvidia-cuda-toolkit
nvcc --version
# nvidia-smi
```


    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Apr_24_19:10:27_PDT_2019
    Cuda compilation tools, release 10.1, V10.1.168



## Compile and run C++ program


```
g++ add.cpp -o add
time ( ./add )
```


## Compiling CUDA versions

Based on https://devblogs.nvidia.com/even-easier-introduction-cuda/

- `nvprof` has fallen out of favor
- `nv-nsight-cu-cli` seems to work in its place

```
nvcc add.cu -o add_cuda
./add_cuda
sudo nv-nsight-cu-cli -o add_cuda ./add_cuda
nv-nsight-cu-cli --import ./add_cuda.nsight-cuprof-report  | grep -i duration

nvcc add_block.cu -o add_block
./add_block
sudo nv-nsight-cu-cli -o add_block ./add_block
nv-nsight-cu-cli --import add_block.nsight-cuprof-report | grep -i duration

nvcc add_grid.cu -o add_grid
./add_grid
sudo nv-nsight-cu-cli -o add_grid ./add_grid
nv-nsight-cu-cli --import add_grid.nsight-cuprof-report | grep -i duration

nvcc add_grid_init.cu -o add_grid_init
./add_grid_init
sudo nv-nsight-cu-cli -o add_grid_init ./add_grid_init
nv-nsight-cu-cli --import add_grid_init.nsight-cuprof-report | grep -i duration

```


|app   | time     | memory tput |
|------|----------|-------------|
|ST    | 56.74 ms |   1.41 GB/s |
|block |  1.26 ms |  10.9  GB/s |
|grid  | 29.47 µs | 421.67 GB/s |



- block: 44.0x speedup over single thread

- grid:  41.7x speedup over single block
  - 1924x speedup over single thread
