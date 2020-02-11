#include <optix.h>
#include <optix_world.h>

#include "../lib/raydata.cuh"

// Sphere variables
rtDeclareVariable(float3, center0, , );
rtDeclareVariable(float3, center1, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(float, time0, , );
rtDeclareVariable(float, time1, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// The point and normal of intersection. and uv-space location
//   the "attribute" qualifier is used to communicate between intersection and shading programs
//   These may only be written between rtPotentialIntersection and rtReportIntersection
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

inline __device__ void get_sphere_uv(const float3 p) {
	float phi = atan2f(p.z, p.x);
	float theta = asinf(p.y);

	hitRecord.u = 1.f - (phi + CUDART_PI_F) / (2.f * CUDART_PI_F);
	hitRecord.v = (theta + CUDART_PIO2_F) / CUDART_PI_F;
}

__device__ float3 center(float time) {
    if (time0 == time1)
        return center0;
    else
        return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

// The sphere bounding box program
//   The pid parameter enables specifying a primitive withing this geometry
//   since there is only 1 primitive (the sphere), the pid is ignored here
RT_PROGRAM void getBounds(int pid, float result[6])
{
    optix::Aabb* box0 = (optix::Aabb*)result;
    box0->m_min = center(time0) - abs(radius);
    box0->m_max = center(time0) + abs(radius);

    optix::Aabb box1;
    box1.m_min = center(time1) - abs(radius);
    box1.m_max = center(time1) + abs(radius);

    box0->include(box1);
}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
RT_PROGRAM void intersection(int pid)
{
    float3 oc = theRay.origin - center(thePrd.gatherTime);
    float a = optix::dot(theRay.direction, theRay.direction);
    float b = optix::dot(oc, theRay.direction);
    float c = optix::dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;

    if (discriminant < 0.0f) return;

    float t = (-b - sqrtf(discriminant)) / a;
    if (t < theRay.tmax && t > theRay.tmin)
        if (rtPotentialIntersection(t))
        {
            hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
            hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD,
                                                                  (hitRecord.point - center(thePrd.gatherTime))/radius));
            get_sphere_uv(hitRecord.normal);
            rtReportIntersection(0);
        }

    t = (-b + sqrtf(discriminant)) / a;
    if (t < theRay.tmax && t > theRay.tmin)
        if (rtPotentialIntersection(t))
        {
            hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
            hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD,
                                                                  (hitRecord.point - center(thePrd.gatherTime))/radius));
            get_sphere_uv(hitRecord.normal);
            rtReportIntersection(0);
        }
}
