#ifndef VECTOR_UTILS_CUH
#define VECTOR_UTILS_CUH

// optix code
#include <optix.h>
#include <optixu/optixu_math_namespace.h>


// if (a < b), return a, else return b
inline __device__ float ffmin(float a, float b) {
	return a < b ? a : b;
}

// if (a > b), return a, else return b
inline __device__ float ffmax(float a, float b) {
	return a > b ? a : b;
}

// return pairwise min vector
inline __device__ float3 min_vec(float3 a, float3 b) {
	return make_float3(ffmin(a.x, b.x), ffmin(a.y, b.y), ffmin(a.z, b.z));
}

// return pairwise max vector
inline __device__ float3 max_vec(float3 a, float3 b) {
	return make_float3(ffmax(a.x, b.x), ffmax(a.y, b.y), ffmax(a.z, b.z));
}

// return max component of vector
inline __device__ float max_component(float3 a){
	return ffmax(ffmax(a.x, a.y), a.z);
}

// return max component of vector
inline __device__ float min_component(float3 a){
	return ffmin(ffmin(a.x, a.y), a.z);
}

#endif //!VECTOR_UTILS_CUH
