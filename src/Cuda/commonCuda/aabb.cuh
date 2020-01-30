#ifndef AABB_CUH
#define AABB_CUH

#include "commonCuda/rtweekend.cuh"

#include <thrust/swap.h>

__forceinline__ __device__ float ffmin(float a, float b) { return a < b ? a : b; }
__forceinline__ __device__ float ffmax(float a, float b) { return a > b ? a : b; }

class aabb {
public:
  __device__ aabb() {}
  __device__ aabb(const vec3& a, const vec3& b) { _min = a; _max = b;}

  __device__ vec3 min() const {return _min; }
  __device__ vec3 max() const {return _max; }

  // __device__ inline bool hit(const ray& r, float tmin, float tmax) const {
  //   for (int a = 0; a < 3; a++) {
  //     float t0 = ffmin((_min[a] - r.origin()[a]) / r.direction()[a],
  //                      (_max[a] - r.origin()[a]) / r.direction()[a]);
  //     float t1 = ffmax((_min[a] - r.origin()[a]) / r.direction()[a],
  //                      (_max[a] - r.origin()[a]) / r.direction()[a]);
  //     tmin = ffmax(t0, tmin);
  //     tmax = ffmin(t1, tmax);
  //     if (tmax <= tmin)
  //       return false;
  //   }
  //   return true;
  // }

  __device__ inline bool hit(const ray& r, float tmin, float tmax) const {
    for (int a = 0; a < 3; a++) {
      float invD = 1.0f / r.direction()[a];
      float t0 = (min()[a] - r.origin()[a]) * invD;
      float t1 = (max()[a] - r.origin()[a]) * invD;
      if (invD < 0.0f)
        thrust::swap(t0, t1);
      tmin = t0 > tmin ? t0 : tmin;
      tmax = t1 < tmax ? t1 : tmax;
      if (tmax <= tmin)
        return false;
    }
    return true;
  }

  vec3 _min;
  vec3 _max;
};

__forceinline__ __device__ aabb surrounding_box(aabb box0, aabb box1) {
  vec3 small( ffmin(box0.min().x(), box1.min().x()),
              ffmin(box0.min().y(), box1.min().y()),
              ffmin(box0.min().z(), box1.min().z()));
  vec3 big  ( ffmax(box0.max().x(), box1.max().x()),
              ffmax(box0.max().y(), box1.max().y()),
              ffmax(box0.max().z(), box1.max().z()));
  return aabb(small,big);
}


#endif
