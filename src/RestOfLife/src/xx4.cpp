#include "random.h"

#include <iostream>

// compute Integral(x^2, 0, 2)
// approximate using:
//     2*average(x^2, 0, 2)    with a randomly sampled x


int main() {
  // int inside_circle = 0;
  // int inside_circle_stratified = 0;
  int N = 1000000;
  double sum;
  for (int i = 0; i < N; i++) {
    double x = 2*random_double();
    sum += x*x;
  }
  std::cout << "I =" << 2*sum/N << "\n";
}
