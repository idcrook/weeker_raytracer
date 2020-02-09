#ifndef RAYDATA_CUH
#define RAYDATA_CUH

#include <optix.h>
// defines CUDART_PI_F
#include "math_constants.h"

typedef enum
{
    Ray_Miss,
    Ray_Hit,
    Ray_Finish,
    Ray_Cancel
} ScatterEvent;

struct PerRayData
{
    float3 attenuation;         // 12 bytes
    float3 scattered_origin;    // 12 bytes
    float3 scattered_direction; // 12 bytes
    float3 emitted;             // 12 bytes
    ScatterEvent scatterEvent;  //  4 bytes
    float  pdf;                 //  4 bytes
    float  scattered_pdf;       //  4 bytes
    float  gatherTime;          //  4 bytes
    uint   seed;                //  4 bytes
};

struct HitRecord
{
    float3 point;  // 12 bytes
    float u;       //  4 bytes
    float v;       //  4 bytes
    float3 normal; // 12 bytes
};

#endif //!RAYDATA_CUH
