#ifndef IO_GEOMETRY_H
#define IO_GEOMETRY_H

#include <optix.h>
#include <optixu/optixpp.h>

class ioGeometry
{
public:
  ioGeometry() { }

  virtual void init(optix::Context& context) = 0;

  virtual void destroy()
    {
      m_geo->destroy();
    }

  optix::Geometry get()
    {
      return m_geo;
    }

protected:
  optix::Geometry m_geo;
};

#endif //!IO_GEOMETRY_H
