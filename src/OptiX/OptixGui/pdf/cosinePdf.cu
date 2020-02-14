
#include "pdf.cuh"

RT_CALLABLE_PROGRAM float3 cosineGenerate(pdf_in &in, uint32_t& seed) {
    float3 temp = randomCosineDirection(seed);
    in.light_direction = in.uvw.local(temp);
    return in.light_direction;
}

RT_CALLABLE_PROGRAM float cosineValue(pdf_in &in) {
    float cosine = optix::dot(optix::normalize(in.direction), in.uvw.w);
    if(cosine > 0.f)
        return cosine / CUDART_PI_F;
    else
        return 0.f;
}
