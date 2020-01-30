#include "random.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

inline double pdf(float x) {
  return 0.5*x;
}

int main() {
  int N = 1000000;
  double sum;
  for (int i = 0; i < N; i++) {
    float x = sqrt(4*random_double());
    sum += x*x / pdf(x);
  }
  std::cout << "I =" << sum/N << "\n";
}
