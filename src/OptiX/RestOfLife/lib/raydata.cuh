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

typedef enum {
    Lambertian,
    DiffuseLight,
    Metallic,
    Dielectric,
    Isotropic,
    Normalic,
} MaterialType;

struct PerRayData
{
    float3 attenuation;         // 12 bytes
    float3 scattered_origin;    // 12 bytes
    float3 scattered_direction; // 12 bytes // obviated by PDF importance sampling for non-specular materials
    float3 emitted;             // 12 bytes
    float3 hit_normal;          // 12 bytes - Need to save per ray?
    ScatterEvent scatterEvent;  //  4 bytes
    float  pdf;                 //  4 bytes
    MaterialType materialType;  //  4 bytes
    float  scattered_pdf;       //  4 bytes
    float  gatherTime;          //  4 bytes
    uint   seed;                //  4 bytes
    bool   is_specular;         //  1 byte?

};

struct HitRecord
{
    float3 point;   // 12 bytes
    float3 normal;  // 12 bytes
    float distance; //  4 bytes
    float u;        //  4 bytes
    float v;        //  4 bytes
};

#endif //!RAYDATA_CUH
