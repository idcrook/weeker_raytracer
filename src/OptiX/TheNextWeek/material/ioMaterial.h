#ifndef IO_MATERIAL_H
#define IO_MATERIAL_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "texture/ioTexture.h"


/*! abstraction for a material that can create, and parameterize,
  a newly created GI's material and closest hit program */
struct ioMaterial {
  virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
};


// class ioMaterial
// {
// public:
//   ioMaterial() { }

//   virtual void init(optix::Context& context) = 0;

//   virtual void destroy()
//     {
//       m_mat->destroy();
//     }

//   optix::Material get()
//     {
//       return m_mat;
//     }

// protected:
//   optix::Material m_mat;
// };

#endif //!IO_MATERIAL_H
