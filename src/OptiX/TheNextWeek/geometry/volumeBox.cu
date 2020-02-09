#include <optix.h>
#include <optix_world.h>

// defines FLT_
#include <cfloat>

#include "../lib/raydata.cuh"
#include "../lib/random.cuh"
#include "../lib/vector_utils.cuh"

/*! the parameters that describe each the box */
rtDeclareVariable(float3, boxMin, , );
rtDeclareVariable(float3, boxMax, , );
rtDeclareVariable(float,  density, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );


#define EPSILON (0.0001f)

// Programs that performs the ray-box intersection
//

inline __device__ bool hit_boundary(const float tMin, const float tMax, float &rec) {
    float3 t0 = (boxMin - theRay.origin) / theRay.direction;
    float3 t1 = (boxMax - theRay.origin) / theRay.direction;
    float temp1 = max_component(min_vec(t0, t1));
    float temp2 = min_component(max_vec(t0, t1));

    if(temp1 > temp2)
        return false;

    // if the first root was a hit,
    if (temp1 < tMax && temp1 > tMin) {
        rec = temp1;
        return true;
    }

    // if the second root was a hit,
    if (temp2 < tMax && temp2 > tMin){
        rec = temp2;
        return true;
    }

    return false;
}


RT_PROGRAM void hitVolume(int pid) {
    float hitt1, hitt2;

    if(hit_boundary(-FLT_MAX, FLT_MAX, hitt1)) {
        if(hit_boundary(hitt1 + EPSILON, FLT_MAX, hitt2)){
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


/*! returns the bounding box of the pid'th primitive
  in this geometry. Since we handle multiple boxes by having a different
  geometry per box, the'pid' parameter is ignored */
RT_PROGRAM void getBounds(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;
    // rtPrintf("boxMin(%f,%f,%f)\n", boxMin.x, boxMin.y, boxMin.z);
    // rtPrintf("boxMax(%f,%f,%f)\n", boxMax.x, boxMax.y, boxMax.z);
    // NOTE: assume all components of boxMin are less than  boxMax
    aabb->m_min = boxMin - make_float3(EPSILON);
    aabb->m_max = boxMax + make_float3(EPSILON);
}
