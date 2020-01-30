#ifndef HITTABLE_CUH     /* -*- cuda -*- */
#define HITTABLE_CUH

#include "commonCuda/rtweekend.cuh"
#include "commonCuda/aabb.cuh"

class material;

struct hit_record
{
  float t;
  vec3 p;
  vec3 normal;
  material *mat_ptr;
};

class hittable  {
public:
  __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
  __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};

#endif
