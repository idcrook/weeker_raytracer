#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "commonCuda/rtweekend.cuh"

class Texture {
public:
  __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public Texture {
public:
  __device__ constant_texture() {}
  __device__ constant_texture(vec3 c) : color(c) {}
  __device__ virtual vec3 value(float u, float v, const vec3& p) const {
    (void)u; (void)v; (void)p;
    return color;
  }
  vec3 color;
};

class checker_texture : public Texture {
public:
  __device__ checker_texture() {}
  __device__ checker_texture(Texture *t0, Texture *t1): even(t0), odd(t1) {}
  __device__ virtual vec3 value(float u, float v, const vec3& p) const {
    float sines = sinf(10.f*p.x())*sinf(10.f*p.y())*sinf(10.f*p.z());
    if (sines < 0.f)
      return odd->value(u, v, p);
    else
      return even->value(u, v, p);
  }
  Texture *even;
  Texture *odd;
};

// class noise_texture : public texture {
// public:
//   noise_texture() {}
//   noise_texture(float sc) : scale(sc) {}
//   virtual vec3 value(float u, float v, const vec3& p) const {
//     (void)u; (void)v;
//     return vec3(1,1,1) * 0.5 * (1 + sin(scale*p.z() + 10*noise.turb(p)));
//   }
//   perlin noise;
//   float scale;
// };

// class image_texture : public texture {
// public:
//   image_texture() {}
//   image_texture(unsigned char *pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
//   virtual vec3 value(float u, float v, const vec3& p) const;
//   unsigned char *data;
//   int nx, ny;
// };

// vec3 image_texture::value(float u, float v, const vec3& p) const {
//   (void)p;
//   int i = (  u)*nx;
//   int j = (1-v)*ny-0.001;
//   if (i < 0) i = 0;
//   if (j < 0) j = 0;
//   if (i > nx-1) i = nx-1;
//   if (j > ny-1) j = ny-1;
//   float r = int(data[3*i + 3*nx*j]  ) / 255.0;
//   float g = int(data[3*i + 3*nx*j+1]) / 255.0;
//   float b = int(data[3*i + 3*nx*j+2]) / 255.0;
//   return vec3(r, g, b);
// }


// #endif

#endif
