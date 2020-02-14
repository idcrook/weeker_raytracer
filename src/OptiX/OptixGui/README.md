Using conan as C++ dependency manager
=====================================

1.	Install conan: https://docs.conan.io/en/latest/installation.html
2.	Clone this repo:
3.	Install dependencies (via `conan`), compile and run

Linux
-----

```bash
cd src/OptiX/OptixGui
mkdir build
cd build

# workaround for busted main repo libX11
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan | true

conan install .. -s build_type=Release
#cmake .. -DCMAKE_BUILD_TYPE=Release
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_FLAGS="--use_fast_math --generate-line-info" \
     -B . ..

conan remote remove bincrafters

cmake --build . --target optixGui --parallel 7

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./optixGui


```

Debug build
-----------

```
conan install .. -s build_type=Debug
# run generate
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CUDA_FLAGS="--use_fast_math --generate-line-info" \
     -B . ..

cmake --build . --target optixGui --parallel 7
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./optixGui

```

#### earlies debugging of trying to remove problematic libGL.so

remove problematic `libGL.so`

```
❯ ./optixGui
MESA-LOADER: failed to open swrast (search paths /home/conan/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/dri)
libGL error: failed to load driver: swrast
Glfw Error 65543: GLX: Failed to create context: BadMatch
❯ LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./optixGui


❯ strace ./optixGui >& out.log || grep GL out.log

rm /home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so*
rm: remove symbolic link '/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so'? y
rm: remove symbolic link '/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so.1'? y
rm: remove regular file '/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/libGL.so.1.2.0'? y

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_FLAGS="--use_fast_math --generate-line-info" \
     -B . ..

...
-- Library GL not found in package, might be system one
...

(now will build and link)

grep -R /lib/libGL.so .

sed -i.bak '/\/lib\/libGL.so/d' ./CMakeFiles/optixGui.dir/build.make
#sed -i.bak 's/[^ ]\+\/lib\/libGL.so//g' ./CMakeFiles/optixGui.dir/link.txt
```

now should run

```
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./optixGui
```
