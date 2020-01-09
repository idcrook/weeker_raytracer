#include "ray.h"

#include <iostream>

bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    // p(t) = A + B*t , circle origin at C (center), radius R (radius)
    // t^2 * dot(B,B) + 2*t*dot(B, A-C) + dot(A-C, A-C) - R^2 = 0
    // a*t^2 + b*t + c
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);
}

vec3 color(const ray& r) {
    if (hit_sphere(vec3(0,0,-2), 0.7071, r))
        return vec3(1, 0, 0);
    vec3 unit_direction = unit_vector(r.direction());
    // scales y [-1, 1] -> t [0, 1]
    float t = 0.5*(unit_direction.y() + 1.0);
    // linear interpolate from white (t = 0) to light blue (t = 1)
    return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

// ./build/apps/program > output/ch5a.ppm

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
