#ifndef IO_MOVING_SPHERE_H
#define IO_MOVING_SPHERE_H

#include "ioGeometry.h"

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char moving_sphere_ptx_c[];

class ioMovingSphere : public ioGeometry
{
public:
  ioMovingSphere()
    {
      m_cx0 = 0.0f;
      m_cy0 = 0.0f;
      m_cz0 = 0.0f;
      m_cx1 = 0.0f;
      m_cy1 = 0.0f;
      m_cz1 = 0.0f;
      m_r = 0.0f;
      m_t0 = 0.0f;
      m_t1 = 0.0f;
    }

  ioMovingSphere(const float x, const float y, const float z,
                 const float x1, const float y1, const float z1,
                 const float r,
                 const float t0 = 0.f, const float t1 = 1.f)
    {
      m_cx0 = x;
      m_cy0 = y;
      m_cz0 = z;
      m_cx1 = x1;
      m_cy1 = y1;
      m_cz1 = z1;
      m_t0 = t0;
      m_t1 = t1;
    }

  virtual void init(optix::Context& context)
    {
      m_geo = context->createGeometry();
      m_geo->setPrimitiveCount(1);
      m_geo->setBoundingBoxProgram(
        context->createProgramFromPTXString(moving_sphere_ptx_c, "getBounds")
        );
      m_geo->setIntersectionProgram(
        context->createProgramFromPTXString(moving_sphere_ptx_c, "intersection")
        );
      m_geo["center0"]->setFloat(m_cx0, m_cy0, m_cz0);
      m_geo["center1"]->setFloat(m_cx1, m_cy1, m_cz1);
      m_geo["radius"]->setFloat(m_r);
      m_geo["time0"]->setFloat(m_t0);
      m_geo["time1"]->setFloat(m_t1);
    }

private:
  float m_cx0;
  float m_cy0;
  float m_cz0;
  float m_cx1;
  float m_cy1;
  float m_cz1;
  float m_r;
  float m_t0;
  float m_t1;
};

#endif //!IO_MOVING_SPHERE_H
