#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "raydata.cuh"

using namespace optix;

// Optix program built-in indices
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

// Ray state variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, world, , );

rtBuffer<float3, 2> resultBuffer;

RT_PROGRAM void rayGenProgram()
{
    float3 lowerLeftCorner = make_float3(-2.0f, -1.0f, -1.0f);
    float3 horizontal = make_float3(4.0f, 0.0f, 0.0f);
    float3 vertical = make_float3(0.0f, 2.0f, 0.0f);
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);

    float u = float(launchIndex.x) / float(launchDim.x);
    float v = float(launchIndex.y) / float(launchDim.y);

    float3 direction = lowerLeftCorner + (u*horizontal) + (v*vertical) - origin;

    optix::Ray ray = optix::make_Ray(
        origin,        // origin
        direction,     // direction
        0,             // raytype
        0.000001f,     // tmin (epsilon)
        RT_DEFAULT_MAX // tmax
        );

    PerRayData prd;
    rtTrace(world, ray, prd);

    float3 drawColor = make_float3(0.0f, 0.0f, 0.0f);

    if (prd.scatterEvent == miss)
    { // Didn't hit anything
        float3 unitDirection = normalize(direction);
        float t = 0.5f * (unitDirection.y + 1.0f);
        drawColor = (1.0f-t) * make_float3(1.0f, 1.0f, 1.0f)
            + t * make_float3(0.5f, 0.7f, 1.0f);
    }
    else
    { // hit something
        drawColor = make_float3(1.0f, 0.0f, 0.0f);
    }

    resultBuffer[launchIndex] = drawColor;
}
