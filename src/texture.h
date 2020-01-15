#ifndef TEXTUREH
#define TEXTUREH

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

#endif
