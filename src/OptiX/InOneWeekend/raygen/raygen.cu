#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "raydata.cuh"
#include "random.cuh"

using namespace optix;

// Optix program built-in indices
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );

// Ray state variables
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, sysWorld, , );
rtBuffer<float3, 2> sysOutputBuffer;

// Ray Generation variables
rtDeclareVariable(int, numSamples, , );

inline __device__ float3 color(optix::Ray& theRay, uint32_t& seed)
{
    PerRayData thePrd;
    rtTrace(sysWorld, theRay, thePrd);

    float3 drawColor = make_float3(0.0f, 0.0f, 0.0f);

    if (thePrd.scatterEvent == miss)
    { // Didn't hit anything
        float3 unitDirection = normalize(theRay.direction);
        float t = 0.5f * (unitDirection.y + 1.0f);
        drawColor = (1.0f-t) * make_float3(1.0f, 1.0f, 1.0f)
            + t * make_float3(0.5f, 0.7f, 1.0f);
    }
    else
    { // hit something
        drawColor = 0.5f * (thePrd.attenuation + make_float3(1.0f,1.0f,1.0f));
    }

    return drawColor;
}

RT_PROGRAM void rayGenProgram()
{
    float3 lowerLeftCorner = make_float3(-2.0f, -1.0f, -1.0f);
    float3 horizontal = make_float3(4.0f, 0.0f, 0.0f);
    float3 vertical = make_float3(0.0f, 2.0f, 0.0f);
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);

    uint32_t seed = tea<64>(theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x, 0);

    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    for (int n = 0; n < numSamples; n++)
    {
        float s = float(theLaunchIndex.x+randf(seed)) / float(theLaunchDim.x);
        float t = float(theLaunchIndex.y+randf(seed)) / float(theLaunchDim.y);
        float3 initialOrigin = origin;
        float3 initialDirection = lowerLeftCorner + (s*horizontal) + (t*vertical) - origin;

        optix::Ray theRay = optix::make_Ray(
            initialOrigin,        // origin
            initialDirection,     // direction
            0,             // raytype
            0.000001f,     // tmin (epsilon)
            RT_DEFAULT_MAX // tmax
            );

        float3 sampleRadiance = color(theRay, seed);
        radiance += sampleRadiance;
    }
    radiance /= numSamples;

    sysOutputBuffer[theLaunchIndex] = radiance;
}
