#ifndef IO_METAL_MATERIAL_H
#define IO_METAL_MATERIAL_H

#include "ioMaterial.h"
#include "texture/ioTexture.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char metal_material_ptx_c[];

class ioMetalMaterial : public ioMaterial
{
public:
  ioMetalMaterial() { }

  ioMetalMaterial(const ioTexture *t,  float fuzz) : texture(t), fuzz(fuzz) {}

  // ioMetalMaterial(float r, float g, float b, float roughness)
  //   : m_r(r), m_g(g), m_b(b), m_roughness(roughness) { }

  virtual void assignTo(optix::GeometryInstance gi, optix::Context& context)  override
    {
      m_mat = context->createMaterial();
      m_mat->setClosestHitProgram(0, context->createProgramFromPTXString
      (metal_material_ptx_c, "closestHit"));
      gi->setMaterial(/*ray type:*/0, m_mat);
      texture->assignTo(gi, context);

      if (fuzz < 1.f) {
          gi["fuzz"]->setFloat(fuzz);
      } else {
          gi["fuzz"]->setFloat(1.f);
      }
    }

private:
  const ioTexture* texture;
   float fuzz;
};

#endif //!IO_METAL_MATERIAL_H
