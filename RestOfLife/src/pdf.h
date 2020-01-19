#ifndef PDFH
#define PDFH

#include "onb.h"
#include "random.h"
#include "vec3.h"


inline vec3 random_cosine_direction() {
  float r1 = random_double();
  float r2 = random_double();
  float z = sqrt(1-r2);
  float phi = 2*M_PI*r1;
  float x = cos(phi)*sqrt(r2);
  float y = sin(phi)*sqrt(r2);
  return vec3(x, y, z);
}


#endif
