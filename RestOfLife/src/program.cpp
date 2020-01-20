#include "float.h"

#include "aarect.h"
#include "box.h"
#include "bvh.h"
#include "hittable_list.h"
#include "material.h"
#include "constant_medium.h"
#include "pdf.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "camera.h"
#include "random.h"
#include "texture.h"
#include "surface_texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

vec3 color(const ray& r, hittable *world, int depth) {
  hit_record rec;
  if (world->hit(r, 0.001, MAXFLOAT, rec)) {
    ray scattered;
    vec3 attenuation;
    vec3 emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
    float pdf_val;
    vec3 albedo;
    if (depth < 50 && rec.mat_ptr->scatter(r, rec, albedo, scattered, pdf_val)) {
      hittable *light_shape = new xz_rect(213, 343, 227, 332, 554, 0);
      hittable_pdf p(light_shape, rec.p);
      scattered = ray(rec.p, p.generate(), r.time());
      pdf_val = p.value(scattered.direction());

      return emitted +
        (albedo*rec.mat_ptr->scattering_pdf(r, rec, scattered)
         * color(scattered, world, depth+1)) / pdf_val;
    }
    else {
      return emitted;
    }
  }
  else {
    return vec3(0,0,0);
  }
}

void cornell_box(hittable **scene, camera **cam, float aspect) {
  int i = 0;
  hittable **list = new hittable*[8];
  material *red = new lambertian( new constant_texture(vec3(0.65, 0.05, 0.05)) );
  material *white = new lambertian( new constant_texture(vec3(0.73, 0.73, 0.73)) );
  material *green = new lambertian( new constant_texture(vec3(0.12, 0.45, 0.15)) );
  material *light = new diffuse_light( new constant_texture(vec3(15, 15, 15)) );
  list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
  list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
  // light source now has a emitting direction
  list[i++] = new flip_normals(new xz_rect(213, 343, 227, 332, 554, light));
  list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
  list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
  list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
  list[i++] = new
    translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18), vec3(130,0,65));
  list[i++] = new
    translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white),  15), vec3(265,0,295));
  *scene = new hittable_list(list,i);
  vec3 lookfrom(278, 278, -800);
  vec3 lookat(278, 278, 0);
  float dist_to_focus = 10.0;
  float aperture = 0.0;
  float vfov = 40.0;
  *cam = new camera(lookfrom, lookat, vec3(0,1,0),
                    vfov, aspect, aperture, dist_to_focus, 0.0, 1.0);
}


// ./build/apps/program > output.ppm

int main (int argc, char** argv) {

  // default values
  bool SUPER_QUALITY_RENDER = !true;
  bool HIGH_QUALITY_RENDER = !true;
  bool MEDIUM_QUALITY_RENDER = !true;

  // handle command line arguments
  if (argc >= 2) {
    // first command line argument is "SH"?
    if (std::string(argv[1]) == "SH") {
      SUPER_QUALITY_RENDER = true;
    }
    // first command line argument is "HQ"?
    if (std::string(argv[1]) == "HQ") {
      HIGH_QUALITY_RENDER = true;
    }
    // first command line argument is "MQ"?
    if (std::string(argv[1]) == "MQ") {
      MEDIUM_QUALITY_RENDER = true;
    }
  }

  int nx, ny, ns;

  if (SUPER_QUALITY_RENDER) {
    nx = 500;
    ny = 500;
    nx *= 2; ny *= 2;
    ns = 10000/4;
  } else if (HIGH_QUALITY_RENDER) {
    nx = 500;
    ny = 500;
    ns = 200;
  } else if (MEDIUM_QUALITY_RENDER) {
    nx = 400;
    ny = 400;
    ns = 24;
  } else {
    nx = 500;
    ny = 500;
    ns = 10;
  }

  std::cerr << "Total Scanlines: " << ny << std::endl;

  std::cout << "P3\n" << nx << " " << ny << "\n255\n";

  hittable *world;
  camera *cam;
  float aspect = float(ny) / float(nx);
  cornell_box(&world, &cam, aspect);


  for (int j = ny-1; j >= 0; j--) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < nx; i++) {
      vec3 col(0, 0, 0);
      for (int s = 0; s < ns; s++) {
        float u = float(i + random_double() - (1.0/2.0)) / float(nx);
        float v = float(j + random_double() - (1.0/2.0)) / float(ny);
        ray r = cam->get_ray(u, v);
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
  std::cerr << "\nDone.\n";

}
