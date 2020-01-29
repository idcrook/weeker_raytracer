#ifndef RAYDATA_CUH
#define RAYDATA_CUH


typedef enum
{
    miss,
    hit
} ScatterEvent;


struct PerRayData
{
    ScatterEvent scatterEvent;
    float3 attenuation;
};


#endif //!RAYDATA_CUH
