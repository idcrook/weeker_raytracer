#ifndef ONB_CUH
#define ONB_CUH

// optix code
#include <optix.h>
#include <optixu/optixu_math_namespace.h>


struct onb {
    // __device__ onb() {}

    __device__ float3 local(float a, float b, float c) const {
        return a*u + b*v + c*w;
    }

    __device__ float3 local(const float3 &a) const {
        return a.x*u + a.y*v + a.z*w;
    }

    __device__ void buildFromW(const float3& n){
        w = optix::normalize(n);

        float3 a;
        if(fabsf(w.x) > 0.9f)
            a = make_float3(0.f, 1.f, 0.f);
        else
            a = make_float3(1.f, 0.f, 0.f);

        v = optix::normalize(optix::cross(w, a));
        u = optix::cross(w, v);
    }

    float3 u, v, w;
};


#endif //!ONB_CUH
