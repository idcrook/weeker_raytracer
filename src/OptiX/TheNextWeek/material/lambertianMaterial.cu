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
rtDeclareVariable(float3, color, , );
// Texture program
//rtDeclareVariable(rtCallableProgramX<float3(float, float, float3)>, constantTexture, ,);
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, constantTexture, , );

RT_PROGRAM void closestHit()
{
    float3 scatterDirection = hitRecord.normal + randomInUnitSphere(thePrd.seed);

    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = color;
    //thePrd.attenuation = constantTexture(0.f, 0.f, hitRecord.point);
}
