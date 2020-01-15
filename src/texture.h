#ifndef TEXTUREH
#define TEXTUREH

#include "perlin.h"
#include "vec3.h"

class texture {
public:
  virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public texture {
public:
  constant_texture() {}
  constant_texture(vec3 c) : color(c) {}
  virtual vec3 value(float u, float v, const vec3& p) const {
    (void)u; (void)v; (void)p;
    return color;
  }
  vec3 color;
};

class checker_texture : public texture {
public:
  checker_texture() {}
  checker_texture(texture *t0, texture *t1): even(t0), odd(t1) {}
  virtual vec3 value(float u, float v, const vec3& p) const {
    float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
    if (sines < 0)
      return odd->value(u, v, p);
    else
      return even->value(u, v, p);
  }
  texture *even;
  texture *odd;
};

class noise_texture : public texture {
public:
  noise_texture() {}
  noise_texture(float sc) : scale(sc) {}
  virtual vec3 value(float u, float v, const vec3& p) const {
    (void)u; (void)v;
    return vec3(1,1,1) * noise.noise(p*scale);
  }
  perlin noise;
  float scale;
};

#endif
