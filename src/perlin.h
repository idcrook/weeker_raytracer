#ifndef PERLINH
#define PERLINH

#include "random.h"
#include "vec3.h"

class perlin {
public:
  float noise(const vec3& p) const {
    float u = p.x() - floor(p.x());
    float v = p.y() - floor(p.y());
    float w = p.z() - floor(p.z());
    int i = floor(p.x());
    int j = floor(p.y());
    int k = floor(p.z());
    return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
  }
  static float *ranfloat;
  static int *perm_x;
  static int *perm_y;
  static int *perm_z;
};

static float* perlin_generate() {
  float * p = new float[256];
  for (int i = 0; i < 256; ++i)
    p[i] = random_double();
  return p;
}

void permute(int *p, int n) {
  for (int i = n-1; i > 0; i--) {
    int target = int(random_double()*(i+1));
    int tmp = p[i];
    p[i] = p[target];
    p[target] = tmp;
  }
  return;
}

static int* perlin_generate_perm() {
  int * p = new int[256];
  for (int i = 0; i < 256; i++)
    p[i] = i;
  permute(p, 256);
  return p;
}

float *perlin::ranfloat = perlin_generate();
int *perlin::perm_x = perlin_generate_perm();
int *perlin::perm_y = perlin_generate_perm();
int *perlin::perm_z = perlin_generate_perm();

#endif
