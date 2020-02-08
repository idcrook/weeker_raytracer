#ifndef IO_GEOMETRY_INSTANCE_H
#define IO_GEOMETRY_INSTANCE_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometry.h"
#include "geometry/ioVolumeBox.h"
#include "geometry/ioVolumeSphere.h"
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

    static optix::GeometryInstance createVolumeBox(const float3& p0, const float3& p1,
                                                   const float density,
                                                   ioMaterial* material, optix::Context &context) {
        ioGeometry* theBoxShape = new ioVolumeBox(p0, p1, density);
        theBoxShape->init(context);
        ioGeometryInstance gi = ioGeometryInstance();
        gi.init(context);
        gi.setGeometry(*theBoxShape);
        material->assignTo(gi.get(), context);

        return gi.get();
    }

    static optix::GeometryInstance createVolumeSphere(const float3& p0, const float radius,
                                                      const float density,
                                                      ioMaterial* material, optix::Context &context) {
        ioGeometry* theSphereShape = new ioVolumeSphere(p0.x, p0.y, p0.z,
                                                        radius, density);
        theSphereShape->init(context);
        ioGeometryInstance gi = ioGeometryInstance();
        gi.init(context);
        gi.setGeometry(*theSphereShape);
        material->assignTo(gi.get(), context);

        return gi.get();
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

private:
    optix::GeometryInstance m_gi;
};

#endif //!IO_GEOMETRY_INSTANCE_H
