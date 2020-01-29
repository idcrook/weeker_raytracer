
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


## runtime error

```
build/src/OptiX/InOneWeekend/inOneWeekendOptix >! output/intro_1.ppm
terminate called after throwing an instance of 'optix::Exception'
  what():  Variable not found (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Variable "Unresolved reference to variable world from _Z13rayGenProgramv" not found in scope)
[1]    19505 abort (core dumped)  build/src/OptiX/InOneWeekend/inOneWeekendOptix >| output/intro_1.ppm


strings build/src/OptiX/InOneWeekend/CMakeFiles/inOneWeekendOptix.dir/cuda_compile_ptx_3_generated_raygen.cu.ptx_embedded.c.o | grep _Z13rayGenProgramv
	// .globl	_Z13rayGenProgramv
.visible .entry _Z13rayGenProgramv(
──────────────────────────────────────────────────────────────
```
