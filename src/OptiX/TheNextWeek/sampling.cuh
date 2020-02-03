#ifndef SAMPLING_CUH
#define SAMPLING_CUH

#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

// defines CUDART_PI_F
#include "math_constants.h"

#include "random.cuh"


inline __device__ float3 random_in_unit_disk(uint32_t& seed) {
    float a = randf(seed) * 2.0f * CUDART_PI_F;

    float3 xy = make_float3(sinf(a), cosf(a), 0);
    xy *= sqrtf(randf(seed));

  return xy;
}


inline __device__ float3 randomInUnitSphere(uint32_t& seed)
{
    float3 p;
    do
    {
        p = 2.0f*make_float3(randf(seed), randf(seed), randf(seed))
            - make_float3(1.0f, 1.0f, 1.0f);
    } while(optix::dot(p,p) >= 1.0f);
    return p;
}

#endif //!SAMPLING_CUH
