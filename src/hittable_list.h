#ifndef HITTABLELISTH
#define HITTABLELISTH

#include "hittable.h"

class hittable_list: public hittable {
    public:
        hittable_list() {}
        hittable_list(hittable **l, int n) {list = l; list_size = n; }
        virtual bool hit(
            const ray& r, float tmin, float tmax, hit_record& rec) const;
        hittable **list;
        int list_size;
};

bool hittable_list::hit(const ray& r, float t_min, float t_max,
                        hit_record& rec) const {

    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif
