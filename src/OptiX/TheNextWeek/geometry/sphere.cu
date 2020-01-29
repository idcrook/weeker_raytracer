#include <optix.h>
#include <optix_world.h>

#include "raydata.cuh"

// Sphere variables
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// The point and normal of intersection
//   the "attribute" qualifier is used to communicate between intersection and shading programs
//   These may only be written between rtPotentialIntersection and rtReportIntersection
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

inline __device__ float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// The sphere bounding box program
//   The pid parameter enables specifying a primitive withing this geometry
//   since there is only 1 primitive (the sphere), the pid is ignored here
RT_PROGRAM void getBounds(int pid, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = center - radius;
    aabb->m_max = center + radius;
}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
RT_PROGRAM void intersection(int pid)
{
    float3 oc = theRay.origin - center;
    float a = dot(theRay.direction, theRay.direction);
    float b = dot(oc, theRay.direction);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;

    if (discriminant < 0.0f) return;

    float t = (-b - sqrtf(discriminant)) / a;
    if (t < theRay.tmax && t > theRay.tmin)
        if (rtPotentialIntersection(t))
        {
            hitRecord.point = theRay.origin + t * theRay.direction;
            hitRecord.normal = (hitRecord.point - center) / radius;
            rtReportIntersection(0);
        }

    t = (-b + sqrtf(discriminant)) / a;
    if (t < theRay.tmax && t > theRay.tmin)
        if (rtPotentialIntersection(t))
        {
            hitRecord.point = theRay.origin + t * theRay.direction;
            hitRecord.normal = (hitRecord.point - center) / radius;
            rtReportIntersection(0);
        }
}
