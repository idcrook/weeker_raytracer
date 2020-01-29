#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <stdint.h>


template <unsigned int N>
static __host__ __device__ __inline__ uint32_t tea(uint32_t s0, uint32_t s1)
{
    uint32_t t = 0;

    for(int n = 0; n < N; n++)
    {
        t += 0x9E3779B9;
        s0 += ((s1 << 4) + 0xa341316c) ^ (s1 + t) ^ ((s1 >> 5) + 0xc8013ea4);
        s1 += ((s0 << 4) + 0xad90777d) ^ (s0 + t) ^ ((s0 >> 5) + 0x7e95761e);
    }      
    return s0;
}


static __host__ __device__ __inline__ uint32_t xorshift32(uint32_t& s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}


static __host__ __device__ __inline__ float randf(uint32_t& s)
{
    float rnd = ((float)xorshift32(s)) / 4294967296.0f;
    if (rnd != 1.0f)
        return rnd;
    else
        return 0x3F7FFFFF;
}

#endif //!RANDOM_CUH
