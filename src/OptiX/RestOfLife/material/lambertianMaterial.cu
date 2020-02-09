#include <optix.h>

#include "material.cuh"

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

__forceinline__ __device__ float3 emitted(){
    return make_float3(0.f, 0.f, 0.f);
}


inline __device__ float scatteringPdf() {
    float cosine = optix::dot(hitRecord.normal, optix::normalize(thePrd.scattered_direction));

    if(cosine < 0.f) {
        cosine = 0.f;
    }

    return cosine / CUDART_PI_F;
}


RT_PROGRAM void closestHit()
{
    float3 scatterDirection = hitRecord.normal + randomInUnitSphere(thePrd.seed);

    thePrd.emitted = emitted();
    thePrd.pdf = optix::dot(hitRecord.normal, scatterDirection) / CUDART_PI_F;
    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.scattered_pdf = scatteringPdf();
    thePrd.attenuation = sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
}
