#include "random.h"
#include "vec3.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

int main() {
    for (int i = 0; i < 200; i++) {
        float r1 = random_double();
        float r2 = random_double();
        float x = cos(2*M_PI*r1)*2*sqrt(r2*(1-r2));
        float y = sin(2*M_PI*r1)*2*sqrt(r2*(1-r2));
        float z = 1 - 2*r2;

        std::cout << x << " " << y << " " << z << "\n";
    }
}
