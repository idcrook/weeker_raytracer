// std
#include <iostream>
#include <time.h>

// optix
#include <optix.h>
#include <optixu/optixpp.h>

// Executive Director
#include "Director.h"

int main(int argc, char* argv[])
{
  clock_t start_time, stop_time, render_time;
  Director optixSingleton = Director();

  start_time = clock();
  optixSingleton.init(1200, 600);
  optixSingleton.createScene();

  render_time = clock();
  optixSingleton.renderFrame();
  stop_time = clock();

  double worldtime_seconds = ((double)(stop_time - start_time)) / CLOCKS_PER_SEC;
  double rendertime_seconds = ((double)(stop_time - render_time)) / CLOCKS_PER_SEC;
  std::cerr << "World took " << worldtime_seconds << " seconds.\n";
  std::cerr << "Raygen took " << rendertime_seconds << " seconds.\n";

  optixSingleton.printPPM();

  optixSingleton.destroy();
}
