# weeker_raytracer

Based on v2 of https://github.com/RayTracing/raytracing.github.io, not the v3 under development

## Build

```
cd TheNextWeek
make clean && make debug
```

## Run

build, then

```
# bang is used here for my zsh setup
time ( build/apps/program >! output/chNa.ppm )
```


## Examples

In One Weekend

Image took about 12.3 minutes, without   BVH. When generating same scene with BVH partitioning, took about 3 minutes.

![final image](img/IOW-ch13f.png)

The Next Week

Image 500x500 with 10,000 ray samples around each point. took 18 hours, 10 minutes

![final image 1](img/TNW-ch10SH2.png)

Image 1000x1000 with 2,500 ray samples around each point. took 18 hours, 14 minutes

![final image 2](img/TNW-ch10gSH.png)
