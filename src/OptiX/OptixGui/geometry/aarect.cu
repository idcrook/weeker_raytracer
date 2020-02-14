#include <optix.h>
#include <optix_world.h>

#include "../lib/raydata.cuh"

/*! the parameters that describe each individual rectangle */
rtDeclareVariable(float,  a0, , );
rtDeclareVariable(float,  a1, , );
rtDeclareVariable(float,  b0, , );
rtDeclareVariable(float,  b1, , );
rtDeclareVariable(float,  k, , );
rtDeclareVariable(int,    flip, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

// Program that performs the ray-sphere intersection
//
// note that this is here is a simple, but not necessarily most numerically
// stable ray-sphere intersection variant out there. There are more
// stable variants out there, but for now let's stick with the one that
// the reference code used.
RT_PROGRAM void hitRectX(int pid) {
    float t = (k - theRay.origin.x) / theRay.direction.x;
    if (t > theRay.tmax || t < theRay.tmin)
        return;

    float a = theRay.origin.y + t * theRay.direction.y;
    float b = theRay.origin.z + t * theRay.direction.z;
    if (a < a0 || a > a1 || b < b0 || b > b1)
        return;

    if (rtPotentialIntersection(t)) {
        hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD, theRay.origin + t * theRay.direction);

        float3 normal = make_float3(1.f, 0.f, 0.f);
        // if (0.f < optix::dot(normal, theRay.direction))
        if (flip)
            normal = -normal;


        hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

        hitRecord.u = (a - a0) / (a1 - a0);
        hitRecord.v = (b - b0) / (b1 - b0);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void hitRectY(int pid) {
    float t = (k - theRay.origin.y) / theRay.direction.y;
    if (t > theRay.tmax || t < theRay.tmin)
        return;

    float a = theRay.origin.x + t * theRay.direction.x;
    float b = theRay.origin.z + t * theRay.direction.z;
    if (a < a0 || a > a1 || b < b0 || b > b1)
        return;

    if (rtPotentialIntersection(t)) {
        hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD, theRay.origin + t * theRay.direction);

        float3 normal = make_float3(0.f, 1.f, 0.f);
        // if (0.f < optix::dot(normal, theRay.direction))
        if (flip)
            normal = -normal;

        hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

        hitRecord.u = (a - a0) / (a1 - a0);
        hitRecord.v = (b - b0) / (b1 - b0);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void hitRectZ(int pid) {
    float t = (k - theRay.origin.z) / theRay.direction.z;
    if (t > theRay.tmax || t < theRay.tmin)
        return;

    float a = theRay.origin.x + t * theRay.direction.x;
    float b = theRay.origin.y + t * theRay.direction.y;
    if (a < a0 || a > a1 || b < b0 || b > b1)
        return;

    if (rtPotentialIntersection(t)) {
        hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD, theRay.origin + t * theRay.direction);

        float3 normal = make_float3(0.f, 0.f, 1.f);
        //if (0.f < optix::dot(normal, theRay.direction))
        if (flip)
            normal = -normal;

        hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

        hitRecord.u = (a - a0) / (a1 - a0);
        hitRecord.v = (b - b0) / (b1 - b0);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void getBoundsX(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    aabb->m_min = make_float3(k - 0.0001f, a0, b0);
    aabb->m_max = make_float3(k + 0.0001f, a1, b1);
}

RT_PROGRAM void getBoundsY(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    aabb->m_min = make_float3(a0, k - 0.0001f, b0);
    aabb->m_max = make_float3(a1, k + 0.0001f, b1);
}

RT_PROGRAM void getBoundsZ(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;

    aabb->m_min = make_float3(a0, b0, k - 0.0001f);
    aabb->m_max = make_float3(a1, b1, k + 0.0001f);
}
