#include "random.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>


int main() {
    int inside_circle = 0;
    int runs = 0;
    while (true) {
        runs++;
        double x = 2*random_double() - 1;
        double y = 2*random_double() - 1;
        if(x*x + y*y < 1)
            inside_circle++;

        if(runs% 100000 == 0)
            std::cout << "\rEstimate of Pi = " << 4*double(inside_circle) / runs << ' ' ;

    }
}
