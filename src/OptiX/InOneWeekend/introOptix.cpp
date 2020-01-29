#include <iostream>

#include <optix.h>
#include <optixu/optixpp.h>

#include "introOptix.h"

extern "C" const char raygen_ptx_c[];
extern "C" const char miss_ptx_c[];
extern "C" const char sphere_ptx_c[];
extern "C" const char material_ptx_c[];

IntroOptix::IntroOptix() {}

void IntroOptix::init(int width, int height)
{
  m_Nx = width;
  m_Ny = height;

  initContext();

  m_context->setEntryPointCount(1);
  initRayGenProgram();
  initMissProgram();

  initOutputBuffer();
  m_context["sysOutputBuffer"]->set(m_outputBuffer);
}

void IntroOptix::destroy()
{
  m_world->destroy();
  m_gi->destroy();
  m_sphere->destroy();
  m_material->destroy();
  m_outputBuffer->destroy();
  m_context->destroy();
}

void IntroOptix::initContext()
{
  m_context = optix::Context::create();
  m_context->setRayTypeCount(1);
}

void IntroOptix::initOutputBuffer()
{
  m_outputBuffer = m_context->createBuffer(RT_BUFFER_OUTPUT);
  m_outputBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_outputBuffer->setSize(m_Nx, m_Ny);
}

void IntroOptix::initRayGenProgram()
{
  m_rayGenProgram = m_context->createProgramFromPTXString(
    raygen_ptx_c, "rayGenProgram");
  m_context->setRayGenerationProgram(0, m_rayGenProgram);
}

void IntroOptix::initMissProgram()
{
  m_missProgram = m_context->createProgramFromPTXString(
    miss_ptx_c, "missProgram");
  m_context->setMissProgram(0, m_missProgram);
}

void IntroOptix::createScene()
{
  // Sphere
  //   Sphere Geometry
  m_sphere = m_context->createGeometry();
  m_sphere->setPrimitiveCount(1);
  m_sphere->setBoundingBoxProgram(
    m_context->createProgramFromPTXString(sphere_ptx_c, "getBounds")
    );
  m_sphere->setIntersectionProgram(
    m_context->createProgramFromPTXString(sphere_ptx_c, "intersection")
    );
  m_sphere["center"]->setFloat(0.0f, 0.0f, -1.0f);
  m_sphere["radius"]->setFloat(0.5f);
  //   Sphere Material
  m_material = m_context->createMaterial();
  optix::Program materialHit = m_context->createProgramFromPTXString(
    material_ptx_c, "closestHit"
    );
  m_material->setClosestHitProgram(0, materialHit);
  //   Sphere GeometryInstance
  m_gi = m_context->createGeometryInstance();
  m_gi->setGeometry(m_sphere);
  m_gi->setMaterialCount(1);
  m_gi->setMaterial(0, m_material);

  // World & Acceleration
  m_world = m_context->createGeometryGroup();
  m_world->setAcceleration(m_context->createAcceleration("Bvh"));
  m_world->setChildCount(1);
  m_world->setChild(0, m_gi);
  // Setting World Variable
  m_context["sysWorld"]->set(m_world);
}

void IntroOptix::renderFrame()
{
  m_context->validate();
  m_context->launch(0,         // Program ID
                    m_Nx, m_Ny // Launch Dimensions
    );
}

void IntroOptix::printPPM()
{
  void* bufferData = m_outputBuffer->map();
  //   Print ppm header
  std::cout << "P3\n" << m_Nx << " " << m_Ny << "\n255\n";
  //   Parse through bufferdata
  for (int j = m_Ny - 1; j >= 0;  j--)
  {
    for (int i = 0; i < m_Nx; i++)
    {
      float* floatData = ((float*)bufferData) + (3*(m_Nx*j + i));
      float r = floatData[0];
      float g = floatData[1];
      float b = floatData[2];
      int ir = int(255.99f * r);
      int ig = int(255.99f * g);
      int ib = int(255.99f * b);
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  m_outputBuffer->unmap();
}

int IntroOptix::getWidth()
{
  return m_Nx;
}
int IntroOptix::getHeight()
{
  return m_Ny;
}
