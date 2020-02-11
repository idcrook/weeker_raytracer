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
        //cosine = 0.f;
        return 0.f;
    } else {
        return cosine / CUDART_PI_F;
    }
}


RT_PROGRAM void closestHit()
{
    onb uvw;
    uvw.buildFromW(hitRecord.normal);
    float3 scatterDirection = optix::normalize(uvw.local(randomCosineDirection(thePrd.seed)));

    thePrd.emitted = emitted();

    thePrd.scatterEvent = Ray_Hit;
    thePrd.hit_normal = hitRecord.normal;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.scattered_pdf = scatteringPdf();
    thePrd.attenuation = sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
    thePrd.pdf = optix::dot(uvw.w, scatterDirection) / CUDART_PI_F;

}
