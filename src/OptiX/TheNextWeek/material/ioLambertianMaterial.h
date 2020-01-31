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

  ioLambertianMaterial(const float r, const float g, const float b)
    : m_r(r), m_g(g), m_b(b) { }

  // ioLambertianMaterial(const ioTexture* t) : texture(t) { }

  // /* create optix material, and assign mat and mat values to geom instance */
  // virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
  //   optix::Material mat = g_context->createMaterial();

  //   mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
  //                             (lambertian_material_ptx_c, "closest_hit"));

  //   gi->setMaterial(/*ray type:*/0, mat);
  //   texture->assignTo(gi, g_context);
  // }

  virtual void init(optix::Context& context) override
    {
      m_mat = context->createMaterial();
      optix::Program hit = context->createProgramFromPTXString(
        lambertian_material_ptx_c, "closestHit"
        );
      hit["color"]->setFloat(m_r, m_g, m_b);
      m_mat->setClosestHitProgram(0, hit);
    }

private:
  float m_r, m_g, m_b;
  // const ioTexture* texture;
};

#endif //!IO_LAMBERTIAN_MATERIAL_H
