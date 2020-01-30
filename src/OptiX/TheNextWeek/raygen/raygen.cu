#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "raydata.cuh"
#include "random.cuh"
#include "camera.cuh"

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

inline __device__ float3 removeNaNs(float3 radiance)
{
    float3 r = radiance;
    if(!(r.x == r.x)) r.x = 0.0f;
    if(!(r.y == r.y)) r.y = 0.0f;
    if(!(r.z == r.z)) r.z = 0.0f;
    return r;
}

inline __device__ float3 missColor(const optix::Ray &theRay)
{
  float3 unitDirection = normalize(theRay.direction);
  float t = 0.5f * (unitDirection.y + 1.0f);
  // "sky" gradient
  float3 missColor = (1.0f-t) * make_float3(1.0f, 1.0f, 1.0f)
      + t * make_float3(0.5f, 0.7f, 1.0f);

  return missColor;
}


inline __device__ float3 color(optix::Ray& theRay, uint32_t& seed)
{
    PerRayData thePrd;
    thePrd.seed = seed;
    float3 sampleRadiance = make_float3(1.0f, 1.0f, 1.0f);
    thePrd.gatherTime = cameraTime0 + randf(seed)*(cameraTime1 - cameraTime0);

    for(int i = 0; i < maxRayDepth; i++)
    {
        rtTrace(sysWorld, theRay, thePrd);
        if (thePrd.scatterEvent == Ray_Miss)
        {
            return sampleRadiance * missColor(theRay);
        }
        else if (thePrd.scatterEvent == Ray_Finish)
        {
            return sampleRadiance * thePrd.attenuation;
        }
        else if (thePrd.scatterEvent == Ray_Cancel)
        {
            return make_float3(0.f);
        }
        else
        {
            // ray is still alive, and got properly bounced
            sampleRadiance *= thePrd.attenuation;
            theRay = optix::make_Ray(
                thePrd.scattered_origin,
                thePrd.scattered_direction,
                0,
                1e-3f,
                RT_DEFAULT_MAX
            );
        }
    }

    seed = thePrd.seed;

    return make_float3(0.0f);
}

RT_PROGRAM void rayGenProgram()
{
    uint32_t seed = tea<64>(theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x, 0);

    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    for (int n = 0; n < numSamples; n++)
    {
        float s = float(theLaunchIndex.x+randf(seed)) / float(theLaunchDim.x);
        float t = float(theLaunchIndex.y+randf(seed)) / float(theLaunchDim.y);
        optix::Ray theRay = generateRay(s,t);
        float3 sampleRadiance = color(theRay, seed);

        // Remove NaNs
        //sampleRadiance = removeNaNs(sampleRadiance);

        radiance += sampleRadiance;
    }
    radiance /= numSamples;

    // gamma correction (2)
    radiance = make_float3(
        sqrtf(radiance.x),
        sqrtf(radiance.y),
        sqrtf(radiance.z)
    );

    sysOutputBuffer[theLaunchIndex] = radiance;
}
