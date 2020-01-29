#include <optix.h>

#include "raydata.cuh"

// Ray state variables
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, sysWorld, , );

// The point and normal of intersection
rtDeclareVariable(float3, hitRecordP, attribute hitRecordP, );
rtDeclareVariable(float3, hitRecordNormal, attribute hitRecordNormal, );


RT_PROGRAM void closestHit()
{
    thePrd.scatterEvent = hit;
    thePrd.attenuation = hitRecordNormal;
}
