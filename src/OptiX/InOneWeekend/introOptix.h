#ifndef INTRO_OPTIX_H
#define INTRO_OPTIX_H

#include <optix.h>
#include <optixu/optixpp.h>

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

  optix::Context m_context;
  optix::Buffer m_outputBuffer;

  optix::Program m_rayGenProgram;
  optix::Program m_missProgram;

  // Scene Objects
  optix::Geometry m_sphere;
  optix::Material m_material;
  optix::GeometryInstance m_gi;
  optix::GeometryGroup m_world;

  void initContext();
  void initOutputBuffer();
  void initRayGenProgram();
  void initMissProgram();
};

#endif //!INTRO_OPTIX_H
