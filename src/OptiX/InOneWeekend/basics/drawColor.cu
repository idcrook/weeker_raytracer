#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float3, 2> resultBuffer;

rtDeclareVariable(float3, color, , );

RT_PROGRAM void drawColor()
{
    resultBuffer[launch_index] = color;
}
