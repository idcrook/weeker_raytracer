C++ package manager
===================

https://docs.conan.io/en/latest/introduction.html

https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html

install conan
-------------

```
sudo apt install ~/projects/learning/rt/github/conan/conan-ubuntu-64_1_22_2.deb
# https://github.com/bincrafters/community/issues/1097
# needed to get glfw dependencies
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
```

```
❯ conan search 'x11*' --remote=conan-center


There are no packages matching the 'x11*' pattern

> conan search 'glf*' --remote=conan-center
Existing package recipes:

glfw/3.2.1@bincrafters/stable
glfw/3.2.1.20180327@bincrafters/stable
glfw/3.3@bincrafters/stable
glfw/3.3.1@bincrafters/stable
glfw/3.3.2@bincrafters/stable
```

example demo
------------

download dependencies with conan

build with cmake

```shell
# needed by meson to build mesa
sudo apt install -y python3-mako
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan | true

cd ~/projects/learning/rt/github/conan/examples/libraries/dear-imgui/basic/build
#conan install .. --build=mesa -s build_type=Release
#cmake .. -DCMAKE_BUILD_TYPE=Release
conan install .. --build=mesa -s build_type=Debug
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build . --clean-first

conan remote remove bincrafters
```

run

```
./dear-imgui-conan
MESA-LOADER: failed to open swrast (search paths /home/conan/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/dri)
libGL error: failed to load driver: swrast
Glfw Error 65543: GLX: Failed to create context: BadMatch

strace ./dear-imgui-conan >& log.out
#  the system one works!
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libGL.so.1", O_RDONLY|O_CLOEXEC) = 3
```

problem with the conan mesa `libGL` for my system, so just delete it

```
# https://askubuntu.com/a/903488

sudo ldconfig -p | grep -i gl.so

# had to delete the libGL.so.1 files in

❯  find /home/dpc/.conan -iname "*libGL.so*" -exec ls -l -- {} +


```

dead-ends Debugging failure
---------------------------

install

```
❯ glxinfo -B

Command 'glxinfo' not found, but can be installed with:

sudo apt install mesa-utils

❯ sudo apt install mesa-utils

❯ sudo apt install libgl1-mesa-glx

❯ find /usr -iname "*libGL.so*" -exec ls -l -- {} +

lrwxrwxrwx 1 root root     14 Mar 13  2019 /usr/lib/i386-linux-gnu/libGL.so.1 -> libGL.so.1.7.0
-rw-r--r-- 1 root root 411160 Mar 13  2019 /usr/lib/i386-linux-gnu/libGL.so.1.7.0
lrwxrwxrwx 1 root root     14 Mar 13  2019 /usr/lib/x86_64-linux-gnu/libGL.so -> libGL.so.1.7.0
lrwxrwxrwx 1 root root     14 Nov 30 19:30 /usr/lib/x86_64-linux-gnu/libGL.so.1 -> libGL.so.1.7.0
-rw-r--r-- 1 root root 596296 Mar 13  2019 /usr/lib/x86_64-linux-gnu/libGL.so.1.7.0

> LIBGL_DEBUG=verbose glxgears

> sudo apt install libglx-mesa0
Reading package lists... Done
Building dependency tree
Reading state information... Done
libglx-mesa0 is already the newest version (19.2.8-0ubuntu0~19.10.2).
libglx-mesa0 set to manually installed.
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.

same for
libgl1-mesa-dri
```

one more try https://github.com/bincrafters/community/issues/1119

```
LIBGL_DRIVERS_PATH=/home/dpc/.conan/data/mesa/19.3.1/bincrafters/stable/package/a56cf85a12b68f87c51b8bc2331fe996caedb686/lib/dri build/dear-imgui-conan
```
