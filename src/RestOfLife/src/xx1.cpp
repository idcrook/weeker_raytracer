#include "random.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

int main() {
    int N = 1000;
    int inside_circle = 0;
    for (int i = 0; i < N; i++) {
        float x = 2*random_double() - 1;
        float y = 2*random_double() - 1;
        if(x*x + y*y < 1)
            inside_circle++;
    }
    std::cout << "Estimate of Pi = " << 4*float(inside_circle) / N << "\n";
}
