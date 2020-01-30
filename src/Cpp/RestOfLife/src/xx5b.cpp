#include "random.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

inline double pdf(float x) {
  return 3*x*x/8;
}

int main() {
  int N = 1;
  double sum;
  for (int i = 0; i < N; i++) {
    double x = pow(8*random_double(), 1./3.);
    sum += x*x / pdf(x);
  }
  std::cout << "I =" << sum/N << "\n";
}
