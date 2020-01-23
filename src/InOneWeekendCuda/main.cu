
#include "commonCuda/rtweekend.h"
#include "sphere.h"
#include "hittable_list.h"

#include <iostream>
#include <time.h>
#include <float.h>

/* Note, doing a straight translation from the original C++ will mean that any
   floating-point constants will be doubles and math on the GPU will be forced
   to be double-precision.  This will hurt our performance unnecessarily.
   Special attention to floating point constants must be taken (e.g. 0.5 ->
   0.5f). */

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ vec3 color(const ray& r, hittable **world) {
  hit_record rec;
  if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
    return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
  }
  else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
  }
}

__global__ void render(vec3 *fb, int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hittable **world) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u*horizontal + v*vertical);
  fb[pixel_index] = color(r, world);
}

__global__ void create_world(hittable **d_list, hittable **d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
    *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *d_world    = new hittable_list(d_list,2);
  }
}

__global__ void free_world(hittable **d_list, hittable **d_world) {
  delete *(d_list);
  delete *(d_list+1);
  delete *d_world;
}

int main() {
  int nx = 1200;
  int ny = 600;
  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx*ny;
  size_t fb_size = num_pixels*sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // make our world of hittables
  hittable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
  hittable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  create_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx/tx+1,ny/ty+1);
  dim3 threads(tx,ty);
  render<<<blocks, threads>>>(fb, nx, ny,
                              vec3(-2.0, -1.0, -1.0),
                              vec3(4.0, 0.0, 0.0),
                              vec3(0.0, 2.0, 0.0),
                              vec3(0.0, 0.0, 0.0),
                              d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j*nx + i;
      int ir = int(255.99*fb[pixel_index].r());
      int ig = int(255.99*fb[pixel_index].g());
      int ib = int(255.99*fb[pixel_index].b());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}
