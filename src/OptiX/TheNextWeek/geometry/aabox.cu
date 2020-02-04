#include <optix.h>
#include <optix_world.h>

#include "../lib/raydata.cuh"

/*! the parameters that describe each the box */
rtDeclareVariable(float3, boxMin, , );
rtDeclareVariable(float3, boxMax, , );

// The ray that will be intersected against
rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePrd, rtPayload,  );

rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

// Programs that performs the ray-box intersection
//
static __device__ float3 boxNormal(float t, float3 t0, float3 t1) {
  float3 neg = make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
  float3 pos = make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
  return pos - neg;
}

// Program that performs the ray-box intersection
RT_PROGRAM void hitBox(int pid) {
    float3 t0 = (boxMin - theRay.origin) / theRay.direction;
    float3 t1 = (boxMax - theRay.origin) / theRay.direction;
    // float tmin = max_component(min_vec(t0, t1));
    // float tmax = min_component(max_vec(t0, t1));
    float tmin = t0.x;
    float tmax = t1.z;

    if(tmin <= tmax) {
      bool check_second = true;

      if(rtPotentialIntersection(tmin)) {
        float3 hit_point = theRay.origin + tmin * theRay.direction;
        hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
        hitRecord.point = hit_point;

        hitRecord.u = 0.f;
        hitRecord.v = 0.f;

        float3 normal = boxNormal(tmin, t0, t1);
        normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hitRecord.normal = normal;

        if(rtReportIntersection(0))
            check_second = false;
      }

      if(check_second) {
        if(rtPotentialIntersection(tmax)) {
            float3 hit_point = theRay.origin + tmax * theRay.direction;
            hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
            hitRecord.point = hit_point;

            hitRecord.u = 0.f;
            hitRecord.v = 0.f;

            float3 normal = boxNormal(tmax, t0, t1);
            normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
            hitRecord.normal = normal;

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
  aabb->m_min = boxMin;
  aabb->m_max = boxMax;
}
