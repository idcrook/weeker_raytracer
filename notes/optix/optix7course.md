# Siggraph 2019 OptiX 7 Course Tutorial Code

found at https://gitlab.com/ingowald/optix7course



```
sudo apt install libglfw3-dev cmake-curses-gui

```


## build

In clone

```shell
# <wherever you installed OptiX 7.0 SDK>
export OptiX_INSTALL_DIR=/usr/local/nvidia/NVIDIA-OptiX-SDK-7.0.0-linux64/

git clone https://gitlab.com/ingowald/optix7course.git
cd optix7course
#mkdir build
#cd build
#make
cmake -B build
cmake --build build
```


## run

```
ls build
build/ex03_testFrameInWindow
```

`ex07` and above require an .obj file to be downloaded and placed in a created `models` directory.  See README

## other

.obj model view

```shell
$ apt-cache search openctm
libopenctm-dev - Library headers for compression of 3D triangle meshes
libopenctm1 - Library for compression of 3D triangle meshes
openctm-doc - Documentation for OpenCTM library and tools
openctm-tools - Tools for compression of 3D triangle meshes
python-openctm - Python bindings for OpenCTM library

$ sudo apt install openctm-tools

$ ctmviewer file.obj
```


others to try

https://geeks3d.com/madview3d/

- g3dviewer- 3D model viewer for GTK+
- mm3d - OpenGL based 3D model editor
  - installs blender and wings3d, rec-s yafaray

```

```

## blender 2.81     - 21 November 2019

```
sudo snap install blender --classic
```

Has OptiX support

### clone blender SVN


```
svn list  https://svn.blender.org/svnroot/bf-blender/trunk/

cd ~/projects/learning/rt/github/models
svn checkout https://svn.blender.org/svnroot/bf-blender/trunk/lib/benchmarks/
mv benchmarks blender_benchmarks

# fun note: blender won't open a file path if it has double underscore in its path components.
```
