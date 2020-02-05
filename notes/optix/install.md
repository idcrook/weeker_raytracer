
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
