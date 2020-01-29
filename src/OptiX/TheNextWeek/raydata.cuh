#ifndef RAYDATA_CUH
#define RAYDATA_CUH

#include <optix.h>

typedef enum
{
    Ray_Miss,
    Ray_Hit,
    Ray_Finish,
    Ray_Cancel
} ScatterEvent;

struct PerRayData
{
    uint seed;
    ScatterEvent scatterEvent;
    optix::Ray scatter;
    float3 attenuation;
};

struct HitRecord
{
    float u;
    float v;
    float3 point;
    float3 normal;
};

#endif //!RAYDATA_CUH
