#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "material.cuh"

// Ray state variables
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// "Global" variables
rtDeclareVariable(rtObject, sysWorld, , );

// The point and normal of intersection and UV parms
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

/*! and finally - that particular material's parameters */
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );


inline __device__ float3 emitted() {

    if(optix::dot(hitRecord.normal, theRay.direction) < 0.f) {
        return sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
    } else {
        return make_float3(0.f);
    }
}

inline __device__ float scatteringPdf() {
    return false;
}

RT_PROGRAM void closestHit() {
    thePrd.emitted = emitted();
    thePrd.hit_normal = hitRecord.normal;
    thePrd.scatterEvent = Ray_Cancel;
    thePrd.scattered_pdf = scatteringPdf();
}
