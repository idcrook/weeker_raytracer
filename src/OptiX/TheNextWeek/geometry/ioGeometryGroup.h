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
      m_gg->setAcceleration(context->createAcceleration("Bvh"));
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
