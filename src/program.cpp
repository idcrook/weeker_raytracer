#include "float.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "camera.h"
#include "random.h"

#include <iostream>

vec3 color(const ray& r, hittable *world, int depth) {
  hit_record rec;
  if (world->hit(r, 0.0001, MAXFLOAT, rec)) {
    ray scattered;
    vec3 attenuation;
    if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return attenuation*color(scattered, world, depth+1);
    }
    else {
      return vec3(0,0,0);
    }
  }
  else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
  }
}
// ./build/apps/program > output/ch12a.ppm

int main() {
  int nx = 200;
  int ny = 100;
  int ns = 100;
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";

  // create hittable list (and add to world)
  hittable *list[5];
  list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
  list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
  list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.3));
  // simulate a bubble with a smaller sphere with a "negative" radius
  list[3] = new sphere(vec3(-1,0,-1), 0.55, new dielectric(1.5));
  list[4] = new sphere(vec3(-1,0,-1), -0.50, new dielectric(1.5));
  hittable *world = new hittable_list(list,5);

  vec3 lookfrom(3,3,2);
  vec3 lookat(0.1,0.1,-1);
  float dist_to_focus = (lookfrom-lookat).length();
  float aperture = 2.0;

  camera cam(lookfrom, lookat, vec3(0,1,0), 20,
             float(nx)/float(ny), aperture, dist_to_focus);

  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {

      vec3 col(0, 0, 0);
      for (int s = 0; s < ns; s++) {
        float u = float(i + random_double() - (1.0/2.0)) / float(nx);
        float v = float(j + random_double() - (1.0/2.0)) / float(ny);
        // float u = float(i + random_double()) / float(nx);
        // float v = float(j + random_double()) / float(ny);
        ray r = cam.get_ray(u, v);
        col += color(r, world, 0);
      }
      col /= float(ns);

      // gamma 2 correction -> pow(1/gamma) aka square root
      col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
}
