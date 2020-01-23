weeker_raytracer
================

Started with v2 of https://github.com/RayTracing/raytracing.github.io here are some examples

#### Rest Of Life

1000x1000 pixels with 500 ray samples around each point. took 1 hour, 8 minutes

![final image](img/ROL-ch13dSH.png)

#### The Next Week

1000x1000 pixels with 2500 ray samples around each point. took over 18 hours

![final image 2](img/TNW-ch10hSH.png)

#### In One Weekend

Image took about 12.3 minutes, without BVH. When generating same scene with BVH partitioning, took about 3 minutes.

![final image](img/IOW-ch13f.png)

Build
-----

-	Using `cmake`
-	Source code needs c++11 compatible compiler

```shell
cmake -B build
cmake --build build
```

-	specify the target with the `--target <program>` option, where the program may be `inOneWeekend`, `theNextWeek`, `restOfLife`

	```
	cmake --build build --target inOneWeekend
	cmake --build build --target theNextWeek
	cmake --build build --target restOfLife
	```

Use `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` so that emacs irony-mode can know the compiler flags

```
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build
```

## Build CUDA

cmake support is included.

Code based on https://github.com/rogerallen/raytracinginoneweekendincuda

```
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -B build
# or target specific SM
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CUDA_FLAGS="-arch=sm_75" -B build
```

	```
	cmake --build build --target inOneWeekendCuda
	```


Run
---

build, then

```bash
# bang is used here for my zsh setup to clobber existing file
time ( build/program >! output/iname.ppm )
```
