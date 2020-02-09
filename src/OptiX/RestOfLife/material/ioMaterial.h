#ifndef IO_MATERIAL_H
#define IO_MATERIAL_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "texture/ioTexture.h"


class ioMaterial
{
public:
    ioMaterial() { }

    virtual void destroy() {
        // in case materials are re-used, only need to be destroyed once
        if (m_mat) {
            m_mat->destroy();
            m_mat = nullptr;
        }
    }

    optix::Material get() {
        return m_mat;
    }

    virtual void assignTo(optix::GeometryInstance gi, optix::Context &context) = 0;

protected:
    optix::Material m_mat;
};

#endif //!IO_MATERIAL_H
