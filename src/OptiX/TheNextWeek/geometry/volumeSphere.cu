#include <optix.h>
#include <optix_world.h>

#include <cfloat>

#include "../lib/raydata.cuh"
#include "../lib/random.cuh"

// Sphere variables
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(float, density, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

// The point and normal of intersection. and uv-space location
//   the "attribute" qualifier is used to communicate between intersection and shading programs
//   These may only be written between rtPotentialIntersection and rtReportIntersection
rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

#define EPSILON (0.0001f)

// assume p is normalized direction vector
inline __device__ void get_sphere_uv(const float3 p) {
	float phi = atan2(p.z, p.x);
	float theta = asin(p.y);

	hitRecord.u = 1 - (phi + CUDART_PI_F) / (2.f * CUDART_PI_F);
	hitRecord.v = (theta + CUDART_PIO2_F) / CUDART_PI_F;
}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
inline __device__ bool hit_boundary(const float tmin, const float tmax, float &rec)
{
    float3 oc = theRay.origin - center;
    float a = optix::dot(theRay.direction, theRay.direction);
    float b = optix::dot(oc, theRay.direction);
    float c = optix::dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;

    if (discriminant < 0.f) return false;

    float t = (-b - sqrtf(discriminant)) / a;
    if (t < tmax && t > tmin) {
        rec = t;
        return true;
    }

    t       = (-b + sqrtf(discriminant)) / a;
    if (t < tmax && t > tmin) {
        rec = t;
        return true;
    }

    return false;
}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
RT_PROGRAM void hitVolume(int pid)
{
    float hitt1, hitt2;


    if(hit_boundary(-FLT_MAX, FLT_MAX, hitt1)) {
        if(hit_boundary(hitt1 + EPSILON, FLT_MAX, hitt2)) {
            if(hitt1 < theRay.tmin)
                hitt1 = theRay.tmin;

            if(hitt2 > theRay.tmax)
                hitt2 = theRay.tmax;

            if(hitt1 >= hitt2)
                return;

            if(hitt1 < 0.f)
                hitt1 = 0.f;


            float distanceInsideBoundary = hitt2 - hitt1;
            distanceInsideBoundary *= optix::length(theRay.direction);

            float hitDistance = -(1.f / density) * logf(randf(thePrd.seed));
            float hitt = hitt1 + (hitDistance / optix::length(theRay.direction));

            if (rtPotentialIntersection(hitt)) {
                hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + hitt*theRay.direction);

                hitRecord.u = 0.f;
                hitRecord.v = 0.f;

                float3 normal = make_float3(1.f, 0.f, 0.f);
                hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

                rtReportIntersection(0);
            }
        }
    }

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
