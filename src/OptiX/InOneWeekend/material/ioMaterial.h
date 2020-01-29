#ifndef IO_MATERIAL_H
#define IO_MATERIAL_H

#include <optix.h>
#include <optixu/optixpp.h>

class ioMaterial
{
public:
  ioMaterial() { }

  virtual void init(optix::Context& context) = 0;

  virtual void destroy()
    {
      m_mat->destroy();
    }

  optix::Material get()
    {
      return m_mat;
    }

protected:
  optix::Material m_mat;
};

#endif //!IO_MATERIAL_H
