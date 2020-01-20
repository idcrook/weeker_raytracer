#ifndef PDFH
#define PDFH

#include "hittable.h"
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

class pdf  {
public:
  virtual float value(const vec3& direction) const = 0;
  virtual vec3 generate() const = 0;
};

class cosine_pdf : public pdf {
public:
  cosine_pdf(const vec3& w) { uvw.build_from_w(w); }
  virtual float value(const vec3& direction) const {
    float cosine = dot(unit_vector(direction), uvw.w());
    if (cosine > 0)
      return cosine/M_PI;
    else
      return 0;
  }
  virtual vec3 generate() const  {
    return uvw.local(random_cosine_direction());
  }
  onb uvw;
};

class hittable_pdf : public pdf {
public:
  hittable_pdf(hittable *p, const vec3& origin) : ptr(p), o(origin) {}
  virtual float value(const vec3& direction) const {
    return ptr->pdf_value(o, direction);
  }
  virtual vec3 generate() const {
    return ptr->random(o);
  }
  hittable *ptr;
  vec3 o;
};


#endif
