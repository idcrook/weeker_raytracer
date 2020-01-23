#ifndef MATERIAL_H             /* -*- cuda -*- */
#define MATERIAL_H

#include "commonCuda/rtweekend.h"
#include "hittable.h"

#include <curand_kernel.h>

struct hit_record;

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
  vec3 p;
  do {
    p = 2.0f*RANDVEC3 - vec3(1,1,1);
  } while (p.squared_length() >= 1.0f);
  return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
  return v - 2.0f*dot(v,n)*n;
}

class material  {
public:
  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
public:
  __device__ lambertian(const vec3& a) : albedo(a) {}
  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target-rec.p);
    attenuation = albedo;
    return true;
  }

  vec3 albedo;
};

class metal : public material {
public:
  __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }
  vec3 albedo;
  float fuzz;
};

#endif
