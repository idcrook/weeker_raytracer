#ifndef DIRECTOR_H
#define DIRECTOR_H

// Include before the optix includes
#include "scene/ioScene.h"

#include <optix.h>
#include <optixu/optixpp.h>


class Director
{
public:
  Director();

  void init(int width, int height);
  void destroy();

  void createScene();
  void renderFrame();
  void printPPM();

  // int getWidth();
  // int getHeight();

private:
  int m_Nx;
  int m_Ny;
  int m_Ns;
  int m_maxRayDepth;

  optix::Context m_context;
  optix::Buffer m_outputBuffer;

  optix::Program m_rayGenProgram;
  optix::Program m_missProgram;
  // optix::Program m_exceptionProgram;

  // Scene Objects
  ioScene m_scene;

  void initContext();
  void initOutputBuffer();
  void initRayGenProgram();
  void initMissProgram();
  // void initExceptionProgram();
};

#endif //!DIRECTOR_H
