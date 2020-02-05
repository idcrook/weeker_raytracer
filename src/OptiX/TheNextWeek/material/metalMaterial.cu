
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "../lib/raydata.cuh"
#include "../lib/sampling.cuh"

// Ray state variables
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, sysWorld, , );

// The point and normal of intersection
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

// Material variables
rtDeclareVariable(float, fuzz, , );

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );

inline __device__ float3 emitted(){
    return make_float3(0.f, 0.f, 0.f);
}

RT_PROGRAM void closestHit()
{
    float3 scatterDirection =
        optix::reflect(theRay.direction, hitRecord.normal)
        + fuzz*randomInUnitSphere(thePrd.seed);

    if (optix::dot(scatterDirection, hitRecord.normal) <= 0.0f )
    { // Ray is absorbed by the material
        thePrd.scatterEvent = Ray_Finish;
        thePrd.scattered_origin = hitRecord.point;
        thePrd.scattered_direction = scatterDirection;
        // no need to calculate if ray absorbed
        thePrd.attenuation = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    thePrd.emitted = emitted();
    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
}
