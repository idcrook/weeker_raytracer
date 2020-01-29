#include <optix.h>

#include "raydata.cuh"

// Ray state variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, world, , );

// The point and normal of intersection
rtDeclareVariable(float3, hitRecordP, attribute hitRecordP, );
rtDeclareVariable(float3, hitRecordNormal, attribute hitRecordNormal, );


RT_PROGRAM void closestHit()
{
    prd.scatterEvent = hit;
    prd.attenuation = hitRecordNormal;
}

