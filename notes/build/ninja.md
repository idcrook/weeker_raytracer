# ninja


https://ninja-build.org/

works with `cmake` as a supported generator


```
sudo apt install ninja-build
cd weeker_raytracer  # clone of this repository
cmake -GNinja -B build
cmake --build build --target inOneWeekend
# or alternatively
cd build
ninja
```
