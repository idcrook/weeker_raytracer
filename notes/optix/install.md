
# Install

- Ubuntu Linux x64 19.10

Downloaded SDKs from Nvidia developer site.

```
# needs to be writeable by user
mkdir -p /usr/local/nvidia/

sh NVIDIA-OptiX-SDK-6.5.0-linux64.sh --skip-license \
  --prefix=/usr/local/nvidia --include-subdir

sh NVIDIA-OptiX-SDK-7.0.0-linux64.sh --skip-license \
   --prefix=/usr/local/nvidia --include-subdir
```


# configure gcc


```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20

sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

Must set to gcc-8 for compiles to work



## install SDK examples

Optix 6.5

```
# there may be other dependencies
sudo apt-get install freeglut3-dev
cd /usr/local/nvidia/NVIDIA-OptiX-SDK-6.5.0-linux64/SDK
cmake -B build .
cmake --build build
```


Optix 7.0

installs, but with warnings about OpenEXR missing

```
Could NOT find OpenEXR (missing: OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE Open
EXR_IlmThread_RELEASE OpenEXR_INCLUDE_DIR) (found version "")
CMake Warning at optixDemandTexture/CMakeLists.txt:62 (message):
  OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
  optixDemandTexture.
-- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11")
-- Found ZlibStatic: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11")
-- Could NOT find OpenEXR (missing: OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE Open
EXR_IlmThread_RELEASE OpenEXR_INCLUDE_DIR) (found version "")
CMake Warning at optixDemandTextureAdvanced/CMakeLists.txt:62 (message):
  OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
  optixDemandTextureAdvanced.
```

build

```
# there may be other dependencies
sudo  apt install openexr openexr-doc openexr-viewers libopenexr-dev
cd /usr/local/nvidia/NVIDIA-OptiX-SDK-7.0.0-linux64/SDK
cmake -B build .
cmake --build build
```

still have openexr related warnings

```
Could NOT find OpenEXR (missing: OpenEXR_INCLUDE_DIR) (found version "")
CMake Warning at optixDemandTexture/CMakeLists.txt:62 (message):
  OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
  optixDemandTexture.
-- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11")
-- Found ZlibStatic: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.11")
-- Could NOT find OpenEXR (missing: OpenEXR_INCLUDE_DIR) (found version "")
CMake Warning at optixDemandTextureAdvanced/CMakeLists.txt:62 (message):
  OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
  optixDemandTextureAdvanced.
```
