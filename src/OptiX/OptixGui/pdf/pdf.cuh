#ifndef PDF_CUH
#define PDF_CUH

#include <optix.h>
#include <optix_world.h>
#include <math_constants.h>

#include "../lib/onb.cuh"
#include "../lib/sampling.cuh"

// communication between hit functions and the value programs
struct pdf_rec {
    float distance;
    float3 normal;
};

// input structs for the PDF programs
struct pdf_in {
    __device__ pdf_in(const float3 o, const float3 d, const float3 n)
                                : origin(o), direction(d), normal(n) {
        uvw.buildFromW(normal);
    }

    const float3 origin;
    const float3 direction;
    const float3 normal;
    float3 light_direction;
    onb uvw;
};


#endif // PDF_CUH
