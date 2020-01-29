#ifndef INTRO_OPTIX_H
#define INTRO_OPTIX_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometry.h"
#include "geometry/ioGeometryInstance.h"
#include "geometry/ioGeometryGroup.h"

class IntroOptix
{
public:
  IntroOptix();

  void init(int width, int height);
  void destroy();

  void createScene();
  void renderFrame();
  void printPPM();

  int getWidth();
  int getHeight();

private:
  int m_Nx;
  int m_Ny;
  int m_Ns;

  optix::Context m_context;
  optix::Buffer m_outputBuffer;

  optix::Program m_rayGenProgram;
  optix::Program m_missProgram;

  // Scene Objects
  ioGeometry* m_pGeometry;
  ioMaterial* m_pMaterial;
  ioGeometryInstance m_gi;
  ioGeometryGroup m_gg;

  void initContext();
  void initOutputBuffer();
  void initRayGenProgram();
  void initMissProgram();
};

#endif //!INTRO_OPTIX_H
