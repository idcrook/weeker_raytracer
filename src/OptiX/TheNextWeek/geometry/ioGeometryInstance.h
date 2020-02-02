#ifndef IO_GEOMETRY_INSTANCE_H
#define IO_GEOMETRY_INSTANCE_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometry.h"
#include "material/ioMaterial.h"
#include "texture/ioTexture.h"

class ioGeometryInstance
{
public:
    ioGeometryInstance() { }

    void init(optix::Context& context)
        {
            m_gi = context->createGeometryInstance();
        }

    void destroy()
        {
            m_gi->destroy();
        }

    optix::GeometryInstance get()
        {
            return m_gi;
        }

    void setGeometry(ioGeometry& geo)
        {
            m_gi->setGeometry(geo.get());
        }

    void setMaterial(ioMaterial& mat, optix::Context& context)
        {
            m_gi->setMaterialCount(1);
            m_gi->setMaterial(0, mat.get());
            mat.assignTo(m_gi, context);

        }

    // void setTexture(ioTexture& tex)
    //     {
    //         tex.assignTo(m_gi, optix::Context &g_context);
    //     }


private:
    optix::GeometryInstance m_gi;
};

#endif //!IO_GEOMETRY_INSTANCE_H
