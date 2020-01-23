#ifndef RTWEEKEND_H /* -*- cuda -*- */
#define RTWEEKEND_H


// #include <cstdlib>
// #include <limits>
// #include <cmath>

// #include <functional>
// #include <random>

// // Utility Functions

// inline double ffmin(double a, double b) { return a <= b ? a : b; }
// inline double ffmax(double a, double b) { return a >= b ? a : b; }

// inline double random_double() {
//     static std::uniform_real_distribution<double> distribution(0.0, 1.0);
//     static std::mt19937 generator;
//     static std::function<double()> rand_generator =
//       std::bind(distribution, generator);
//     return rand_generator();
//   }

// Common Headers

#include "commonCuda/vec3.h"
#include "commonCuda/ray.h"

#endif
