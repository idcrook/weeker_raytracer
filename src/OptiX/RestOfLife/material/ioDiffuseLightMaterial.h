#ifndef IO_DIFFUSE_LIGHT_H
#define IO_DIFFUSE_LIGHT_H

#include "ioMaterial.h"
#include "texture/ioTexture.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char diffuse_light_material_ptx_c[];


class ioDiffuseLightMaterial : public ioMaterial
{
public:
    ioDiffuseLightMaterial() { }

    ioDiffuseLightMaterial(const ioTexture* t) : texture(t) {}

    virtual void assignTo(optix::GeometryInstance gi, optix::Context& context)  override {
        m_mat = context->createMaterial();
        m_mat->setClosestHitProgram(0, context->createProgramFromPTXString
                                    (diffuse_light_material_ptx_c, "closestHit"));

        gi->setMaterialCount(1);
        gi->setMaterial(/*ray type:*/0, m_mat);
        texture->assignTo(gi, context);
    }


private:
    const ioTexture* texture;

};

#endif //!IO_DIFFUSE_LIGHT_MATERIAL_H
