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
rtDeclareVariable(int, maxRayDepth, , );

inline __device__ float3 color(optix::Ray& theRay, uint32_t& seed)
{
    PerRayData thePrd;
    thePrd.seed = seed;
    float3 sampleRadiance = make_float3(1.0f, 1.0f, 1.0f);

    for(int i = 0; i < maxRayDepth; i++)
    {
        rtTrace(sysWorld, theRay, thePrd);
        if (thePrd.scatterEvent == Ray_Miss)
        {
            float3 unitDirection = normalize(theRay.direction);
            float t = 0.5f * (unitDirection.y + 1.0f);
            float3 missColor = (1.0f-t) * make_float3(1.0f, 1.0f, 1.0f)
                + t * make_float3(0.5f, 0.7f, 1.0f);
            return sampleRadiance * missColor;
        }
        else if (thePrd.scatterEvent == Ray_Finish)
        {
            return sampleRadiance * thePrd.attenuation;
        }
        else if (thePrd.scatterEvent == Ray_Cancel)
        {
            return make_float3(1000000.0f, 0.0f, 1000000.0f);
        }
        else
        {
            // Must have hit something
            sampleRadiance *= thePrd.attenuation;
            theRay = optix::make_Ray(
                thePrd.scatter.origin,
                thePrd.scatter.direction,
                0,
                0.001f,
                RT_DEFAULT_MAX
                );
        }
    }

    seed = thePrd.seed;

    return make_float3(0.0f, 0.0f, 0.0f);
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
        float u = float(theLaunchIndex.x+randf(seed)) / float(theLaunchDim.x);
        float v = float(theLaunchIndex.y+randf(seed)) / float(theLaunchDim.y);
        float3 initialOrigin = origin;
        float3 initialDirection = lowerLeftCorner + (u*horizontal) + (v*vertical) - origin;

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
