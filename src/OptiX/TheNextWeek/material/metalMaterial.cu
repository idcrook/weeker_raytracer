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

RT_PROGRAM void closestHit()
{
    float3 scatterDirection = optix::reflect(theRay.direction, hitRecord.normal) +
        roughness*randomInUnitSphere(thePrd.seed);

    if (optix::dot(scatterDirection, hitRecord.normal) <= 0.0f )
    { // Ray is absorbed by the material
        thePrd.scatterEvent = Ray_Finish;
        thePrd.attenuation = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    thePrd.scatterEvent = Ray_Hit;
    thePrd.scatter = optix::make_Ray(
        hitRecord.point,
        scatterDirection,
        theRay.ray_type,
        theRay.tmin,
        theRay.tmax
        );
    thePrd.attenuation = color;
}
