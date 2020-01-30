#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

rtDeclareVariable(float3, cameraOrigin, , );
rtDeclareVariable(float3, cameraU, , );
rtDeclareVariable(float3, cameraV, , );
rtDeclareVariable(float3, cameraW, , );
rtDeclareVariable(float, cameraTime0, , );
rtDeclareVariable(float, cameraTime1, , );
rtDeclareVariable(float, cameraHalfHeight, , );
rtDeclareVariable(float, cameraHalfWidth, , );
rtDeclareVariable(int, cameraType, , );


__device__ void perspectiveCamera(const float s, const float t,
    float3& origin, float3& direction)
{
    float3 lowerLeftCorner = cameraOrigin
        - cameraHalfWidth*cameraU - cameraHalfHeight*cameraV - cameraW;
    float3 horizontal = 2.0f*cameraHalfWidth*cameraU;
    float3 vertical = 2.0f*cameraHalfHeight*cameraV;

    origin = cameraOrigin;
    direction = lowerLeftCorner + (s*horizontal) + (t*vertical) - cameraOrigin;
}

__device__ void environmentCamera(const float s, const float t,
    float3& origin, float3& direction)
{
    float2 d = make_float2(s*2.0f*3.14159f, t*3.14159f);
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
    float3 lowerLeftCorner = cameraOrigin
        - cameraHalfWidth*cameraU - cameraHalfHeight*cameraV - cameraW;
    float3 horizontal = 2.0f*cameraHalfWidth*cameraU;
    float3 vertical = 2.0f*cameraHalfHeight*cameraV;

    origin = lowerLeftCorner + (s*horizontal) + (t*vertical) + cameraOrigin;
    direction = -optix::normalize(cameraW);
}

inline __device__ optix::Ray generateRay(float s, float t)
{
    float3 initialOrigin, initialDirection;
    if (cameraType == 0)
        perspectiveCamera(s, t, initialOrigin, initialDirection);
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
