#include "ray.h"

#include <iostream>


float hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    // p(t) = A + B*t , circle origin at C (center), radius R (radius)
    // t^2 * dot(B,B) + 2*t*dot(B, A-C) + dot(A-C, A-C) - R^2 = 0
    // a*t^2 + b*t + c
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    if (discriminant < 0) {
      return -1.0;
    } else {
      // use nearest quadratic solution (-z is the viewport direction)
      return (-b - sqrt(discriminant)) / (2.0*a);
    }
}

vec3 color(const ray& r) {
  float t = hit_sphere(vec3(0,0,-1), 0.5, r);
  if (t>0) {
    // compute a unit normal for the sphere origin: (0,0,-1) hit
    vec3 N = unit_vector(r.point_at_parameter(t) - vec3(0,0,-1));
    // scale [-1,1] -> [0,1] and use that as colorspace
    return 0.5*vec3(N.x()+1, N.y()+1, N.z()+1);
  }
  vec3 unit_direction = unit_vector(r.direction());
  // scales y [-1, 1] -> t [0, 1]
  t = 0.5*(unit_direction.y() + 1.0);
  // linear interpolate from white (t = 0) to light blue (t = 1)
  return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

// ./build/apps/program > output/ch6a.ppm

int main() {
    int nx = 200;
    int ny = 100;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            float u = float(i) / float(nx);
            float v = float(j) / float(ny);
            ray r(origin, lower_left_corner + u*horizontal + v*vertical);
            vec3 col = color(r);
            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);

            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}
