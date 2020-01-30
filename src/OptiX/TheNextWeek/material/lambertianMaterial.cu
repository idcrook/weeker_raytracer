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

RT_PROGRAM void closestHit()
{
    float3 scatterDirection = hitRecord.normal + randomInUnitSphere(thePrd.seed);

    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = color;
}
