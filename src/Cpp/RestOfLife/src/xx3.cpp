#include "random.h"

#include <iostream>

int main() {
  int inside_circle = 0;
  int inside_circle_stratified = 0;
  int sqrt_N = 10000;
  for (int i = 0; i < sqrt_N; i++) {
    for (int j = 0; j < sqrt_N; j++) {
      float x = 2*random_double() - 1;
      float y = 2*random_double() - 1;
      if (x*x + y*y < 1)
        inside_circle++;
      x = 2*((i + random_double()) / sqrt_N) - 1;
      y = 2*((j + random_double()) / sqrt_N) - 1;
      if (x*x + y*y < 1)
        inside_circle_stratified++;
    }
  }
  std::cout << "Regular    Estimate of Pi = " <<
    4*float(inside_circle) / (sqrt_N*sqrt_N) << "\n";
  std::cout << "Stratified Estimate of Pi = " <<
    4*float(inside_circle_stratified) / (sqrt_N*sqrt_N) << "\n";
}
