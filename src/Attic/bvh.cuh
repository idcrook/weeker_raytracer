#ifndef BVH_CUH
#define BVH_CUH

#include "commonCuda/rtweekend.cuh"
#include "hittable.cuh"

#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

class bvh_node : public hittable  {
public:
  __device__ bvh_node() {}
  //__device__ bvh_node(hittable **l, int n, float time0, float time1);
  __device__ bvh_node(hittable **l, int n, float time0, float time1, curandState *local_rand_state);
  __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
  __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
  hittable *left;
  hittable *right;
  aabb box;
};


__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
  (void)t0;
  (void)t1;

  b = box;
  return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  if (box.hit(r, t_min, t_max)) {
    hit_record left_rec, right_rec;
    bool hit_left = left->hit(r, t_min, t_max, left_rec);
    bool hit_right = right->hit(r, t_min, t_max, right_rec);
    if (hit_left && hit_right) {
      if (left_rec.t < right_rec.t)
        rec = left_rec;
      else
        rec = right_rec;
      return true;
    }
    else if (hit_left) {
      rec = left_rec;
      return true;
    }
    else if (hit_right) {
      rec = right_rec;
      return true;
    }
    else
      return false;
  }
  else return false;
}

struct BoxXCmp
{
  __device__ bool operator()(const hittable* lhs, const hittable* rhs) const
  {
    aabb box_left, box_right;
    hittable *ah = *(hittable**)lhs;
    hittable *bh = *(hittable**)rhs;
    if(!ah->bounding_box(0,0, box_left) || !bh->bounding_box(0,0, box_right))
      void();
    if ( box_left.min().x() - box_right.min().x() < 0.0f )
      return false;
    else
      return true;
  }
};

struct BoxYCmp
{
  __device__ bool operator()(const hittable* lhs, const hittable* rhs) const
  {
    aabb box_left, box_right;
    hittable *ah = *(hittable**)lhs;
    hittable *bh = *(hittable**)rhs;
    if(!ah->bounding_box(0,0, box_left) || !bh->bounding_box(0,0, box_right))
      void();
    if ( box_left.min().y() - box_right.min().y() < 0.0f )
      return false;
    else
      return true;
  }
};

struct BoxZCmp
{
  __device__ bool operator()(const hittable* lhs, const hittable* rhs) const
  {
    aabb box_left, box_right;
    hittable *ah = *(hittable**)lhs;
    hittable *bh = *(hittable**)rhs;
    if(!ah->bounding_box(0,0, box_left) || !bh->bounding_box(0,0, box_right))
      void();
    if ( box_left.min().z() - box_right.min().z() < 0.0f )
      return false;
    else
      return true;
  }
};


__device__ bvh_node::bvh_node(hittable **l, int n, float time0, float time1, curandState *local_rand_state) {
  int axis = 0;
  //int axis = int(3*curand_uniform(local_rand_state));
  // int axis = int(3*double_random()));
  // if (axis == 0)
  //   qsort(l, n, sizeof(hittable *), box_x_compare);
  // else if (axis == 1)
  //   qsort(l, n, sizeof(hittable *), box_y_compare);
  // else
  //   qsort(l, n, sizeof(hittable *), box_z_compare);
  if (axis == 0)
    thrust::sort(l.begin(), l.end(), BoxXCmp());
  // else if (axis == 1)
  //   qsort(l, n, sizeof(hittable *), box_y_compare);
  // else
  //   qsort(l, n, sizeof(hittable *), box_z_compare);
  if (n == 1) {
    left = right = l[0];
  }
  else if (n == 2) {
    left = l[0];
    right = l[1];
  }
  else {
    left = new bvh_node(l, n/2, time0, time1, local_rand_state);
    right = new bvh_node(l + n/2, n - n/2, time0, time1, local_rand_state);
  }
  aabb box_left, box_right;
  // if(!left->bounding_box(time0,time1, box_left) || !right->bounding_box(time0,time1, box_right))
  //   std::cerr << "no bounding box in bvh_node constructor\n";
  box = surrounding_box(box_left, box_right);
}



#endif
