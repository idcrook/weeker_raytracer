# Optix advanced examples

Repo
https://github.com/nvpro-samples/optix_advanced_samples.git

"Intro"

https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction


## copied over up-to-date optix .cmake files

```diff
+++ b/src/CMakeLists.txt

@@ -199,8 +199,16 @@ if( APPLE )
 endif()


+# set to match your path
+set(OptiX_INSTALL_DIR "/usr/local/nvidia/NVIDIA-OptiX-SDK-6.5.0-linux64/"
+  CACHE PATH "Path to OptiX installed location.")
+
+# Defines OptiX utilities and variables
+INCLUDE(configure_optix)
+
+
 # Search for the OptiX libraries and include files.
-find_package(OptiX REQUIRED)
+#find_package(OptiX REQUIRED)
```


## actual readme

https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction

 also https://gitlab.com/ingowald/optix7course

## install devil

installing `dev` also installs the lib

```
sudo apt install libdevil-dev
```


## building

warning about OpenGL

```
CMake Warning (dev) at /usr/share/cmake-3.13/Modules/FindOpenGL.cmake:270 (message):
   Policy CMP0072 is not set: FindOpenGL prefers GLVND by default when
   available.  Run "cmake --help-policy CMP0072" for policy details.  Use the
   cmake_policy command to set the policy and suppress this warning.

   FindOpenGL found both a legacy GL library:

     OPENGL_gl_LIBRARY: /usr/lib/x86_64-linux-gnu/libGL.so

   and GLVND libraries for OpenGL and GLX:

     OPENGL_opengl_LIBRARY: /usr/lib/x86_64-linux-gnu/libOpenGL.so
     OPENGL_glx_LIBRARY: /usr/lib/x86_64-linux-gnu/libGLX.so

   OpenGL_GL_PREFERENCE has not been set to "GLVND" or "LEGACY", so for
   compatibility with CMake 3.10 and below the legacy GL library will be used.
 Call Stack (most recent call first):
   CMakeLists.txt:132 (find_package)
 This warning is for project developers.  Use -Wno-dev to suppress it.
```

install

```
sudo apt install libglfw3-dev
```

new error

```
CMake Error at support/glfw/CMakeLists.txt:236 (message):
  The Xinerama library and headers were not found
```

install

```
sudo apt install libxinerama-dev
```

new error

```
CMake Error at support/glfw/CMakeLists.txt:267 (message):
  The Xcursor libraries and headers were not found
```

install

```
sudo apt install libxcursor-dev
```


success!

```
cmake -B build src
...
-- Build files have been written to
```

now do actual build

```
cmake --build build
```

Another error

```
In file included from /home/dpc/projects/learning/rt/github/optix/optix_advanced_samples/src/sutil/sutil.cpp:32:
/home/dpc/projects/learning/rt/github/optix/optix_advanced_samples/src/sutil/GL/glew.h:1202:14: fatal error: GL/glu.h: No such file or directory
 #    include <GL/glu.h>
              ^~~~~~~~~~
compilation terminated.
make[2]: *** [sutil/CMakeFiles/sutil_sdk.dir/build.make:194: sutil/CMakeFiles/sutil_sdk.dir/sutil.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:1210: sutil/CMakeFiles/sutil_sdk.dir/all] Error 2
make: *** [Makefile:130: all] Error 2
```

install

```
sudo apt install libglu1-mesa-dev
```

build again

```
cmake -B build src
cmake --build build
```


# Run


```shell
ls build/bin/
build/bin/optixHello
``
