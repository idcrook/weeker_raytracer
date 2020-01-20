# weeker_raytracer

Based on v2 of https://github.com/RayTracing/raytracing.github.io here are some examples

#### Rest Of Life

1000x1000 pixels with 500 ray samples around each point. took 1 hour, 8 minutes

![final image](img/ROL-ch13dSH.png)

#### The Next Week


1000x1000 pixels with 2500 ray samples around each point. took over 18 hours

![final image 2](img/TNW-ch10hSH.png)


#### In One Weekend

Image took about 12.3 minutes, without   BVH. When generating same scene with BVH partitioning, took about 3 minutes.

![final image](img/IOW-ch13f.png)


## Build

```shell
cd ProjectSubDirectory
make clean && make debug
```

## Run

build, then

```shell
# bang is used here for my zsh setup
time ( build/apps/program >! output/chNa.ppm )
```
