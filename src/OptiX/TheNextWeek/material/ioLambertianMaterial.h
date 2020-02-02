#ifndef IO_LAMBERTIAN_MATERIAL_H
#define IO_LAMBERTIAN_MATERIAL_H

#include "ioMaterial.h"
#include "texture/ioTexture.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char lambertian_material_ptx_c[];

class ioLambertianMaterial : public ioMaterial
{
public:
  ioLambertianMaterial() { }

  // ioLambertianMaterial(const float r, const float g, const float b)
  //   : m_r(r), m_g(g), m_b(b) { }

  ioLambertianMaterial(const ioTexture* t) : texture(t) {}

  // virtual void init(optix::Context& context) 
  //   {
  //     m_mat = context->createMaterial();
  //     optix::Program hit = context->createProgramFromPTXString(
  //       lambertian_material_ptx_c, "closestHit"
  //       );
  //     hit["color"]->setFloat(m_r, m_g, m_b);
  //     m_mat->setClosestHitProgram(0, hit);
  //   }

  virtual void assignTo(optix::GeometryInstance gi, optix::Context& context)  override
    {
      m_mat = context->createMaterial();
      m_mat->setClosestHitProgram(0, context->createProgramFromPTXString
      (lambertian_material_ptx_c, "closestHit"));
      gi->setMaterial(/*ray type:*/0, m_mat);
      texture->assignTo(gi, context);
    }


private:
  const ioTexture* texture;

  ///float m_r, m_g, m_b;
};

#endif //!IO_LAMBERTIAN_MATERIAL_H
