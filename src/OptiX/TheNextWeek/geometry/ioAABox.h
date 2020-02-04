#ifndef IO_AA_BOX_H
#define IO_AA_BOX_H

#include "ioGeometry.h"
#include "material/ioMaterial.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char aabox_ptx_c[];

class ioAABox : public ioGeometry
{
public:
    ioAABox() {}

    ioAABox(const float3 p0, const float3 p1)
        {
            m_boxMin = p0;
            m_boxMax = p1;
        }

    void init(optix::Context& context) {
        m_geo = context->createGeometry();
        m_geo->setPrimitiveCount(1);

        m_geo->setBoundingBoxProgram(context->createProgramFromPTXString(aabox_ptx_c, "getBounds"));
        m_geo->setIntersectionProgram(context->createProgramFromPTXString(aabox_ptx_c, "hitBox"));
        _init();
    }

    void _init() {
        m_geo["boxMin"]->setFloat(m_boxMin);
        m_geo["boxMax"]->setFloat(m_boxMax);
    }

private:
    float3 m_boxMin;
    float3 m_boxMax;
};

#endif //!IO_AA_BOX_H
