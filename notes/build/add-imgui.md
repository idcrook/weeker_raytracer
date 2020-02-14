Adding Dear ImGUI
=================

will try [conan](https://docs.conan.io/en/latest/introduction.html) for dependency management

### Get started

start by copying over

```
cd src/OptiX
cp -r RestOfLife OptixGui
# ... update CMakeLists.txt files to correspond, naming executable optixGui

# run generate and confirm it still builds
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CUDA_FLAGS="--use_fast_math --generate-line-info" \
     -B build src
cmake --build build --target optixGui --parallel 7
```
