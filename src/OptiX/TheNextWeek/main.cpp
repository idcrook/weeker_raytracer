// std
#include <iostream>

// optix
#include <optix.h>
#include <optixu/optixpp.h>

// Executive Director
#include "Director.h"

int main(int argc, char* argv[])
{
  Director optixSingleton = Director();
  optixSingleton.init(1200, 600);

  optixSingleton.createScene();

  optixSingleton.renderFrame();
  optixSingleton.printPPM();

  optixSingleton.destroy();
}
