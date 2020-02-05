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

virtual void assignTo(optix::GeometryInstance gi, optix::Context& context)  override
    {
      m_mat = context->createMaterial();
      gi->setMaterialCount(1);
      m_mat->setClosestHitProgram(0, context->createProgramFromPTXString(
        dielectric_material_ptx_c, "closestHit"
        ));
      gi->setMaterial(/*ray type:*/0, m_mat);
      gi["eta"]->setFloat(m_eta);
    }

private:
  float m_eta;
};

#endif //!IO_DIELECTRIC_MATERIAL_H
