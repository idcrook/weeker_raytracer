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

remove problematic `libGL.so`

```
rm /home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so*
rm: remove symbolic link '/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so'? y
rm: remove symbolic link '/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so.1'? y
rm: remove regular file '/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so.1.2.0'? y
```

now should run

```

```
