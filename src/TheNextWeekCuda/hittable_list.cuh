#ifndef HITTABLE_LIST_CUH        /* -*- cuda -*- */
#define HITTABLE_LIST_CUH

#include "hittable.cuh"

class hittable_list: public hittable  {
public:
  __device__ hittable_list() {}
  __device__ hittable_list(hittable **l, int n) {list = l; list_size = n; }
  __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
  __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

  hittable **list;
  int list_size;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  hit_record temp_rec;
  bool hit_anything = false;
  float closest_so_far = t_max;
  for (int i = 0; i < list_size; i++) {
    if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

__device__ bool hittable_list::bounding_box(float t0, float t1, aabb& box) const {
  if (list_size < 1) return false;
  aabb temp_box;
  bool first_true = list[0]->bounding_box(t0, t1, temp_box);
  if (!first_true)
    return false;
  else
    box = temp_box;
  for (int i = 1; i < list_size; i++) {
    if(list[i]->bounding_box(t0, t1, temp_box)) {
      box = surrounding_box(box, temp_box);
    }
    else
      return false;
  }
  return true;
}

#endif
