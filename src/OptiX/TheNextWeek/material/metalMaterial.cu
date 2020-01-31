#include <optix.h>
#include <optixu/optixu_math_namespace.h>

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
rtDeclareVariable(float, roughness, , );

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );


RT_PROGRAM void closestHit()
{
    float3 scatterDirection = optix::reflect(theRay.direction, hitRecord.normal) +
        roughness*randomInUnitSphere(thePrd.seed);

    if (optix::dot(scatterDirection, hitRecord.normal) <= 0.0f )
    { // Ray is absorbed by the material
        thePrd.scatterEvent = Ray_Finish;
        // thePrd.scattered_origin = hitRecord.point;
        // thePrd.scattered_direction = scatterDirection;
        thePrd.attenuation = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = color;
    //thePrd.attenuation = sampleTexture(0.f, 0.f, hitRecord.point);
}
