#ifndef HITTABLE_H     /* -*- cuda -*- */
#define HITTABLE_H

#include "commonCuda/rtweekend.h"

struct hit_record
{
  float t;
  vec3 p;
  vec3 normal;
};

class hittable  {
public:
  __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
