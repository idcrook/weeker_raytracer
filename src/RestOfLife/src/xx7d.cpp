#include "random.h"
#include "vec3.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

inline vec3 random_cosine_direction() {
    float r1 = random_double();
    float r2 = random_double();
    float z = sqrt(1-r2);
    float phi = 2*M_PI*r1;
    float x = cos(phi)*sqrt(r2);
    float y = sin(phi)*sqrt(r2);
    return vec3(x, y, z);
}

int main() {
    int N = 1000000;
    float sum = 0.0;

    for (int i = 0; i < N; i++) {
      vec3 v = random_cosine_direction();
       sum += v.z()*v.z()*v.z() / (v.z()/(M_PI));
    }
    std::cout << "PI/2 = " << M_PI/2 << "\n";
    std::cout << "Estimate = " << sum/N << "\n";
}
