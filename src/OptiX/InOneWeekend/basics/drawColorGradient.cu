#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

rtBuffer<float3, 2> resultBuffer;

rtDeclareVariable(float3, topLeftColor, , );
rtDeclareVariable(float3, bottomRightColor, , );

RT_PROGRAM void drawColorGradient()
{
    float2 ratio = make_float2(launchIndex) / make_float2(launchDim);
    ratio.y = 1.0f - ratio.y;
    float r = bottomRightColor.x * ratio.x + topLeftColor.x * (1.0f - ratio.x) ;
    float g = bottomRightColor.y * ratio.y + topLeftColor.y * (1.0f - ratio.y) ;
    float b = bottomRightColor.z * length(ratio) + topLeftColor.z * (1.0f - length(ratio));
    float3 drawColor = make_float3(r, g, b);
    resultBuffer[launchIndex] = drawColor;
}
