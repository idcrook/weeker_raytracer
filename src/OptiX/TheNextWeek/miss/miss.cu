#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../lib/raydata.cuh"

using namespace optix;

rtDeclareVariable(PerRayData, thePrd, rtPayload, );

RT_PROGRAM void missProgram()
{
    thePrd.scatterEvent = Ray_Miss;
}
