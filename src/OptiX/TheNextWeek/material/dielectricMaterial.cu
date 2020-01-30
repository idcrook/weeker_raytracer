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
rtDeclareVariable(float, eta, , );

inline __device__ float fresnelSchlick(
    const float cosThetaI, const float etaI, const float etaT)
{
    float r0 = (etaI-etaT) / (etaI+etaT);
    r0 = r0*r0;
    return r0 + (1.f-r0)*powf((1.f-cosThetaI), 5.f);
}

RT_PROGRAM void closestHit()
{
    // Get the ray's unit direction
    float3 unitDirection = optix::normalize(theRay.direction);

    // Determine if inside or outside of object
    float3 localNormal;
    float etaI, etaT;
    if (optix::dot(theRay.direction, hitRecord.normal) < 0.0f)
    {
        // Outside the object
        localNormal = hitRecord.normal;
        etaI = 1.0f;
        etaT = eta;
    }
    else
    {
        // Inside the object
        localNormal = -hitRecord.normal;
        etaI = eta;
        etaT = 1.0f;
    }

    // Snell's Law
    //  etaI * sinThetaI = etaT * sinThetaT
    // If
    //  (etaI/etaT) * sinThetaI > 1.0
    // Then
    //  sinThetaT does not exist, and no transmission is possible
    float3 scatterDirection;
    float cosThetaI =
        optix::min(optix::dot(-unitDirection, localNormal), 1.0f);
    float sinThetaI = sqrtf(1.0f - cosThetaI*cosThetaI);
    if ( ((etaI/etaT)*sinThetaI) > 1.0f )
    {
        // No Transmission is possible
        scatterDirection = optix::reflect(unitDirection, localNormal);
    }
    else
    {
        // Transmission + Reflection
        float reflectProb = fresnelSchlick(cosThetaI, etaI, etaT);
        if (randf(thePrd.seed) < reflectProb)
        { // Reflection
            scatterDirection = optix::reflect(unitDirection, localNormal);
        }
        else
        { // Transmission
            float sinThetaT = optix::min((etaI/etaT)*sinThetaI, 1.0f);
            float cosThetaT = sqrtf(1.0f - sinThetaT*sinThetaT);
            scatterDirection =
                (etaI/etaT)*(unitDirection + cosThetaI*localNormal) -
                cosThetaT*localNormal;
        }
    }

    if(cosThetaI > 1.0f)
        printf("costThetaI is greater than unity: %f", cosThetaI);
    if(!(sinThetaI == sinThetaI))
        printf("sinThetaI is NaN: %f", sinThetaI);

    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.attenuation = make_float3(1.0f, 1.0f, 1.0f);
    // thePrd.attenuation = make_float3(0.8f, 0.85f, 0.82f); // for greenish glass
}
