#ifndef IO_VOLUME_SPHERE_H
#define IO_VOLUME_SPHERE_H

#include "ioGeometry.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char volume_sphere_ptx_c[];

class ioVolumeSphere : public ioGeometry
{
public:
    ioVolumeSphere() {
        m_cx0 = 0.0f;
        m_cy0 = 0.0f;
        m_cz0 = 0.0f;
        m_r = 0.0f;
        m_density = 0.0f;
    }

    ioVolumeSphere(const float x, const float y, const float z, const float r, const float density)
        : m_cx0(x), m_cy0(y), m_cz0(z), m_r(r), m_density(density) {}

    virtual void init(optix::Context& context) {
        m_geo = context->createGeometry();
        m_geo->setPrimitiveCount(1);
        m_geo->setBoundingBoxProgram(
            context->createProgramFromPTXString(volume_sphere_ptx_c, "getBounds")
            );
        m_geo->setIntersectionProgram(
            context->createProgramFromPTXString(volume_sphere_ptx_c, "hitVolume")
            );
        m_geo["center"]->setFloat(m_cx0, m_cy0, m_cz0);
        m_geo["radius"]->setFloat(m_r);
        m_geo["density"]->setFloat(m_density);
    }

private:
    float m_cx0;
    float m_cy0;
    float m_cz0;
    float m_r;
    float m_density;
};

#endif //!IO_VOLUME_SPHERE_H
