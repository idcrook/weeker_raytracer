#ifndef IO_GEOMETRY_GROUP_H
#define IO_GEOMETRY_GROUP_H

#include <optix.h>
#include <optixu/optixpp.h>

#include "geometry/ioGeometryInstance.h"

class ioGeometryGroup
{
public:
  ioGeometryGroup() { }

  void init(optix::Context& context)
    {
      m_gg = context->createGeometryGroup();
      // NoAccel, Bvh, Sbvh, Trbvh
      //m_gg->setAcceleration(context->createAcceleration("NoAccel")); // faster than Trbvh with early empty Cornell box
      m_gg->setAcceleration(context->createAcceleration("Trbvh"));
      m_gg->setChildCount(0);
    }

  void destroy()
    {
      m_gg->destroy();
    }

  void addChild(ioGeometryInstance& gi)
    {
      m_gg->addChild(gi.get());
    }

  optix::GeometryGroup get()
    {
      return m_gg;
    }

private:
  optix::GeometryGroup m_gg;
};

#endif //!IO_GEOMETRY_GROUP_H
