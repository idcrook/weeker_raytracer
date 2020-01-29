#ifndef IO_METAL_MATERIAL_H
#define IO_METAL_MATERIAL_H

#include "ioMaterial.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char metal_material_ptx_c[];

class ioMetalMaterial : public ioMaterial
{
public:
  ioMetalMaterial() { }

  ioMetalMaterial(float r, float g, float b, float roughness)
    : m_r(r), m_g(g), m_b(b), m_roughness(roughness) { }

  virtual void init(optix::Context& context) override
    {
      m_mat = context->createMaterial();
      optix::Program hit = context->createProgramFromPTXString(
        metal_material_ptx_c, "closestHit"
        );
      hit["color"]->setFloat(m_r, m_g, m_b);
      hit["roughness"]->setFloat(m_roughness);
      m_mat->setClosestHitProgram(0, hit);
    }

private:
  float m_r, m_g, m_b;
  float m_roughness;
};

#endif //!IO_METAL_MATERIAL_H
