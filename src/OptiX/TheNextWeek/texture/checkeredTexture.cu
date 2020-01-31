
#include "texture.cuh"



rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, odd, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, even, , );

RT_CALLABLE_PROGRAM float3 sampleTexture(float u, float v, float3 p) {
    float sines = sinf(10.f * p.x) * sinf(10.f - p.y) * sinf(10.f * p.z);

    if (sines < 0)
        return odd(u, v, p);
    else
        return even(u, v, p);
}
