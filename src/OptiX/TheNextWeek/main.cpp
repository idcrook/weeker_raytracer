// std
#include <iostream>
#include <chrono>

// Image I/O
// must #define STB_IMAGE_IMPLEMENTATION and do it only once (e.g. in .cpp file)
#define STB_IMAGE_IMPLEMENTATION
#include "../external/rtw_stb_image.h"

// optix
#include <optix.h>
#include <optixu/optixpp.h>

// Executive Director
#include "Director.h"

int main(int argc, char* argv[])
{
  Director optixSingleton = Director();

  auto start = std::chrono::system_clock::now();
  optixSingleton.init(1200, 600);
  optixSingleton.createScene();
  optixSingleton.renderFrame();
  auto stop = std::chrono::system_clock::now();
  auto time_seconds = std::chrono::duration<float>(stop - start).count();
  std::cerr << "Took " << time_seconds << " seconds.\n";

  optixSingleton.printPPM();

  optixSingleton.destroy();
}
