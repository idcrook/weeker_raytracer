#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1

#include <optix.h>

#include "raydata.cuh"
#include "sampling.cuh"

// Ray state variables
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, sysWorld, , );

// The point and normal of intersection
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

// Material variables

// Texture program
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );

inline __device__ float3 emitted(){
    return make_float3(0.f, 0.f, 0.f);
}

RT_PROGRAM void closestHit()
{
    float3 scatterDirection = hitRecord.normal + randomInUnitSphere(thePrd.seed);

    thePrd.emitted = emitted();
    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
}
