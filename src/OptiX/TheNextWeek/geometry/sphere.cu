#include <optix.h>
#include <optix_world.h>

#include "../lib/raydata.cuh"

// Sphere variables
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// The point and normal of intersection. and uv-space location
//   the "attribute" qualifier is used to communicate between intersection and shading programs
//   These may only be written between rtPotentialIntersection and rtReportIntersection
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

// assume p is normalized direction vector
inline __device__ void get_sphere_uv(const float3 p) {
	float phi = atan2f(p.z, p.x);
	float theta = asinf(p.y);

	hitRecord.u = 1 - (phi + CUDART_PI_F) / (2.f * CUDART_PI_F);
	hitRecord.v = (theta + CUDART_PIO2_F) / CUDART_PI_F;
}

// The sphere bounding box program
//   The pid parameter enables specifying a primitive withing this geometry
//   since there is only 1 primitive (the sphere), the pid is ignored here
RT_PROGRAM void getBounds(int pid, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = center - abs(radius);
    aabb->m_max = center + abs(radius);
}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
RT_PROGRAM void intersection(int pid)
{
    float3 oc = theRay.origin - center;
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
            hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, (hitRecord.point - center)/radius));
            get_sphere_uv(hitRecord.normal);
            rtReportIntersection(0);
        }

    t = (-b + sqrtf(discriminant)) / a;
    if (t < theRay.tmax && t > theRay.tmin)
        if (rtPotentialIntersection(t))
        {
            hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
            hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, (hitRecord.point - center)/radius));
            get_sphere_uv(hitRecord.normal);
            rtReportIntersection(0);
        }
}
