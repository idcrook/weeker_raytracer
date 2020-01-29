#include <iostream>

#include <optix.h>
#include <optixu/optixpp.h>

#include "introOptix.h"

extern "C" const char raygen_ptx_c[];
extern "C" const char miss_ptx_c[];

IntroOptix::IntroOptix() {}

void IntroOptix::init(int width, int height)
{
  m_Nx = width;
  m_Ny = height;
  m_Ns = 1024;
  m_maxRayDepth = 50;

  initContext();

  m_context->setEntryPointCount(1);
  initRayGenProgram();
  initMissProgram();

  initOutputBuffer();
  m_context["sysOutputBuffer"]->set(m_outputBuffer);
}

void IntroOptix::destroy()
{
  m_scene.destroy();
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
  m_context["numSamples"]->setInt(m_Ns);
  m_context["maxRayDepth"]->setInt(m_maxRayDepth);
}

void IntroOptix::initMissProgram()
{
  m_missProgram = m_context->createProgramFromPTXString(
    miss_ptx_c, "missProgram");
  m_context->setMissProgram(0, m_missProgram);
}

void IntroOptix::createScene()
{
  m_scene.init(m_context);
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
