#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

// defines CUDART_PI_F
#include "math_constants.h"
#include "../lib/sampling.cuh"

rtDeclareVariable(float3, cameraOrigin, , );
rtDeclareVariable(float3, cameraU, , );
rtDeclareVariable(float3, cameraV, , );
rtDeclareVariable(float3, cameraW, , );
rtDeclareVariable(float, cameraTime0, , );
rtDeclareVariable(float, cameraTime1, , );
rtDeclareVariable(float, cameraLensRadius, , );

rtDeclareVariable(float3, cameraLowerLeftCorner, , );
rtDeclareVariable(float3, cameraHorizontal, , );
rtDeclareVariable(float3, cameraVertical, , );

rtDeclareVariable(int, cameraType, , );


__device__ void perspectiveCamera(const float s, const float t, uint32_t& seed,
    float3& origin, float3& direction)
{
    const float3 rd = cameraLensRadius * random_in_unit_disk(seed);
    const float3 lens_offset = cameraU*rd.x + cameraV*rd.y;
    origin = cameraOrigin + lens_offset;
    direction = cameraLowerLeftCorner + (s*cameraHorizontal) + (t*cameraVertical) - origin;
}

__device__ void environmentCamera(const float s, const float t,
    float3& origin, float3& direction)
{
    float2 d = make_float2(s*2.0f*CUDART_PI_F, t*CUDART_PI_F);
    float3 angle = make_float3(
        cosf(d.x) * sinf(d.y),
        -cosf(d.y),
        sinf(d.x) * sinf(d.y)
    );

    origin = cameraOrigin;
    direction = optix::normalize(
        angle.x*cameraU + angle.y*cameraV + angle.z*cameraW
    );
}

__device__ void orthographicCamera(const float s, const float t,
    float3& origin, float3& direction)
{
    origin = cameraLowerLeftCorner + (s*cameraHorizontal) + (t*cameraVertical) + cameraOrigin;
    direction = -optix::normalize(cameraW);
}

inline __device__ optix::Ray generateRay(float s, float t, uint32_t& seed)
{
    float3 initialOrigin, initialDirection;
    if (cameraType == 0)
        perspectiveCamera(s, t, seed, initialOrigin, initialDirection);
    else if (cameraType == 1)
        environmentCamera(s, t, initialOrigin, initialDirection);
    else if (cameraType == 2)
        orthographicCamera(s, t, initialOrigin, initialDirection);

    return optix::make_Ray(
        initialOrigin,        // origin
        initialDirection,     // direction
        0,                    // raytype
        1e-6f,                // tmin (epsilon)
        RT_DEFAULT_MAX        // tmax
    );
}

#endif //!CAMERA_CUH
