
#include "pdf.cuh"

rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, p0_value, , );
rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, p1_value, , );

rtDeclareVariable(float, bias, , );

RT_CALLABLE_PROGRAM float mixture_value(pdf_in &in) {
    return 0.5f * p0_value(in) + 0.5f * p1_value(in);
}

rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, uint32_t&)>, p0_generate, , );
rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, uint32_t&)>, p1_generate, , );

RT_CALLABLE_PROGRAM float3 mixture_generate(pdf_in &in, uint32_t& seed) {
    if (randf(seed) < bias)
        return p0_generate(in, seed);
    else
        return p1_generate(in, seed);
}
