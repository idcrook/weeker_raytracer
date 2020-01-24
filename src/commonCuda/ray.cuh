#ifndef RAY_CUH  /* -*- cuda -*- */
#define RAY_CUH

#include "commonCuda/vec3.cuh"

class ray
{
public:
  __device__ ray() {}
  __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
  __device__ vec3 origin() const       { return A; }
  __device__ vec3 direction() const    { return B; }
  __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

  vec3 A;
  vec3 B;
};

#endif
