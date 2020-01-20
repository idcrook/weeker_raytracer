#ifndef MATERIALH
#define MATERIALH

#include "hittable.h"
#include "random.h"
#include "texture.h"
#include "onb.h"
#include "pdf.h"

vec3 random_in_unit_sphere() {
  vec3 p;
  do {
    // pick a random point in unit square {x,y,z} in range -1 to +1
    p = 2.0*vec3(random_double(), random_double(), random_double()) - vec3(1,1,1);
  } while (p.squared_length() >= 1.0); // must be within unit sphere
  return p;
}

// angle of incidence is the angle of reflection
// n is unit normal vector
vec3 reflect(const vec3& v, const vec3& n) {
  return v - 2*dot(v,n)*n;
}

// Snell's law: n_0 sin(theta_0) = n_1 sin(theta_1)
// n_0: originating material index of refraction
// n_1: transmitting material index of refraction
// theta_0, theta_1, angle of ray w.r.t. normal
// (n_0 / n_1) * sin(theta_0) = sin (theta_1)
// if n_0 > n_1, there may be no real solution (total internal reflection)
// theta_1 = arcsin( (n_0/ n_1) * sin(theta_0)
// "critical angle" is when theta_1 = 90°

// a = cos(theta_0)*v
// x = sin(theta_0)*v
// sin(theta_0) = x / sqrt(a^2 + x^2)
// cos(theta_0) = dot(uv, n)

// n is unit normal vector
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
  vec3 uv = unit_vector(v);
  float dt = dot(uv, n);
  // TBD: prove these derivations
  float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
  if (discriminant > 0) {
    refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
    return true;
  }
  else
    return false;
}

float schlick(float cosine, float ref_idx) {
  float r0 = (1-ref_idx) / (1+ref_idx);
  r0 = r0*r0;
  return r0 + (1-r0)*pow((1 - cosine),5);
}


class material  {
public:
  virtual bool scatter(const ray& r_in,
                       const hit_record& rec, vec3& albedo, ray& scattered, float& pdf) const {
    (void)r_in; (void)rec;  (void)albedo;  (void)scattered;  (void)pdf;
    return false;
  }

  virtual float scattering_pdf(const ray& r_in, const hit_record& rec,
                               const ray& scattered) const {
    (void)r_in; (void)rec; (void)scattered;
    return 0;
  }

  // not a pure virtual function
  virtual vec3 emitted(const ray& r_in, const hit_record& rec,
                       float u, float v, const vec3& p) const {
    (void)r_in; (void)rec; (void)u; (void)v; (void)p;
      return vec3(0,0,0);
  }
};


class lambertian : public material {
public:
  lambertian(texture *a) : albedo(a) {}

  float scattering_pdf(const ray& r_in,
                       const hit_record& rec, const ray& scattered) const {
    (void)r_in;
    float cosine = dot(rec.normal, unit_vector(scattered.direction()));
    if (cosine < 0)
      return 0;
    return cosine / M_PI;
  }

  bool scatter(
               const ray& r_in,
               const hit_record& rec,
               vec3& alb,
               ray& scattered,
               float& pdf) const
  {
    onb uvw;
    uvw.build_from_w(rec.normal);
    vec3 direction = uvw.local(random_cosine_direction());
    scattered = ray(rec.p, unit_vector(direction), r_in.time());
    alb = albedo->value(rec.u, rec.v, rec.p);
    pdf = dot(uvw.w(), scattered.direction()) / M_PI;
    return true;
  }


  texture *albedo;
};


class metal : public material {
public:
  metal(const vec3& a, float f) : albedo(a) {
    if (f < 1) fuzz = f; else fuzz = 1;
  }
  virtual bool scatter(const ray& r_in, const hit_record& rec,
                       vec3& attenuation, ray& scattered) const {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(), r_in.time());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }
  vec3 albedo;
  float fuzz;
};


class dielectric : public material {
public:
  dielectric(float ri) : ref_idx(ri) {}
  virtual bool scatter(const ray& r_in, const hit_record& rec,
                       vec3& attenuation, ray& scattered) const {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;

    float reflect_prob;
    float cosine;

    if (dot(r_in.direction(), rec.normal) > 0) {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx;
      cosine = ref_idx * dot(r_in.direction(), rec.normal)
        / r_in.direction().length();
    }
    else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0 / ref_idx;
      cosine = -dot(r_in.direction(), rec.normal)
        / r_in.direction().length();
    }

    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
      reflect_prob = schlick(cosine, ref_idx);
    }
    else {
      reflect_prob = 1.0;
    }

    if (random_double() < reflect_prob) {
      scattered = ray(rec.p, reflected, r_in.time());
    }
    else {
      scattered = ray(rec.p, refracted, r_in.time());
    }

    return true;
  }

  float ref_idx;
};


// add an emissive material
class diffuse_light : public material {
public:
  diffuse_light(texture *a) : emit(a) {}
  virtual bool scatter(const ray& r_in, const hit_record& rec,
                       vec3& attenuation, ray& scattered) const {
    (void)r_in; (void)rec; (void)attenuation; (void)scattered;
    return false;
  }

  virtual vec3 emitted(const ray& r_in, const hit_record& rec,
                       float u, float v, const vec3& p) const {
    if (dot(rec.normal, r_in.direction()) < 0.0)
      return emit->value(u, v, p);
    else
      return vec3(0,0,0); // do not emit both down and up (normal dot component > 0)
  }

  texture *emit;
};


class isotropic : public material {
public:
  isotropic(texture *a) : albedo(a) {}
  virtual bool scatter(const ray& r_in,
                       const hit_record& rec,
                       vec3& attenuation,
                       ray& scattered) const {
    (void)r_in;
    scattered = ray(rec.p, random_in_unit_sphere());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }
  texture *albedo;
};


#endif
