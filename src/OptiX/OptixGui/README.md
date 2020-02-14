Using conan as C++ dependency manager
=====================================

1.	Install conan: https://docs.conan.io/en/latest/installation.html
2.	Clone this repo:
3.	Install dependencies, compile and run

Linux
-----

```bash
cd src/OptiX/OptixGui
mkdir build
cd build

conan install .. -s build_type=Release
#cmake .. -DCMAKE_BUILD_TYPE=Release
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_FLAGS="--use_fast_math --generate-line-info" \
     -B . ..

# conan install .. -s build_type=Debug
# # run generate
# #cmake .. -DCMAKE_BUILD_TYPE=Debug
# Debug target seems broken
# cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug \
#      -DCMAKE_CUDA_FLAGS="--use_fast_math --generate-line-info" \
#      -B . ..

cmake --build . --target optixGui --parallel 7
```
