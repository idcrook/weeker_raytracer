#ifndef IO_DIELECTRIC_MATERIAL_H
#define IO_DIELECTRIC_MATERIAL_H

#include "ioMaterial.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char dielectric_material_ptx_c[];

class ioDielectricMaterial : public ioMaterial
{
public:
  ioDielectricMaterial() { }

  ioDielectricMaterial(float eta) : m_eta(eta) { }

  virtual void init(optix::Context& context) override
    {
      m_mat = context->createMaterial();
      optix::Program hit = context->createProgramFromPTXString(
        dielectric_material_ptx_c, "closestHit"
        );
      hit["eta"]->setFloat(m_eta);
      m_mat->setClosestHitProgram(0, hit);
    }

private:
  float m_eta;
};

#endif //!IO_DIELECTRIC_MATERIAL_H
