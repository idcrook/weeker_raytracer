
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "material.cuh"

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

inline __device__ float scatteringPdf() {
  return false;
}

RT_PROGRAM void closestHit()
{
    thePrd.emitted = emitted();

    // optix::reflect expects normalized (unit vector) inputs
    float3 reflected = optix::reflect(optix::normalize(theRay.direction), hitRecord.normal);
    float3 scatterDirection = reflected + fuzz*randomInUnitSphere(thePrd.seed);

    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
    thePrd.scattered_pdf = scatteringPdf();

    if (optix::dot(scatterDirection, hitRecord.normal) <= 0.0f ) {
        thePrd.scatterEvent = Ray_Cancel;
        return;
    }

    thePrd.scatterEvent = Ray_Hit;
}
