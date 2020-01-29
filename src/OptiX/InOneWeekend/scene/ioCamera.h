#ifndef IO_CAMERA_H
#define IO_CAMERA_H

#include <optix.h>

class ioCamera
{
public:
  ioCamera() {}
  virtual ~ioCamera() {}

  virtual void destroy() = 0;

  virtual void init(optix::Context& context) = 0;
};


class ioPerspectiveCamera : public ioCamera
{
public:
  ioPerspectiveCamera(
    float fromX, float fromY, float fromZ,
    float toX, float toY, float toZ,
    float upX, float upY, float upZ,
    float vFov, float aspect
    )
    {
      float theta = vFov * 3.14159f/180.0f;
      m_halfHeight = tan(theta/2.0f);
      m_halfWidth = aspect * m_halfHeight;
      m_origin = optix::make_float3(fromX, fromY, fromZ);
      m_w = optix::normalize(
        optix::make_float3(fromX-toX, fromY-toY, fromZ-toZ)
        );
      m_u = optix::normalize(
        optix::cross(make_float3(upX, upY, upZ), m_w)
        );
      m_v = optix::cross(m_w, m_u);
    }

  virtual void init(optix::Context& context)
    {
      context["cameraOrigin"]->set3fv((float*)&m_origin);
      context["cameraU"]->set3fv((float*)&m_u);
      context["cameraV"]->set3fv((float*)&m_v);
      context["cameraW"]->set3fv((float*)&m_w);
      context["cameraHalfHeight"]->setFloat(m_halfHeight);
      context["cameraHalfWidth"]->setFloat(m_halfWidth);
      context["cameraType"]->setInt(0);
    }

  virtual void destroy() { }

private:
  optix::float3 m_origin;
  optix::float3 m_u;
  optix::float3 m_v;
  optix::float3 m_w;
  float m_halfHeight;
  float m_halfWidth;
};

class ioEnvironmentCamera : public ioCamera
{
public:
  ioEnvironmentCamera(
    float fromX, float fromY, float fromZ,
    float toX, float toY, float toZ,
    float upX, float upY, float upZ
    )
    {
      m_origin = optix::make_float3(fromX, fromY, fromZ);
      m_w = optix::normalize(
        optix::make_float3(fromX-toX, fromY-toY, fromZ-toZ)
        );
      m_u = optix::normalize(
        optix::cross(make_float3(upX, upY, upZ), m_w)
        );
      m_v = optix::cross(m_w, m_u);
    }

  virtual void init(optix::Context& context)
    {
      context["cameraOrigin"]->set3fv((float*)&m_origin);
      context["cameraU"]->set3fv((float*)&m_u);
      context["cameraV"]->set3fv((float*)&m_v);
      context["cameraW"]->set3fv((float*)&m_w);
      context["cameraType"]->setInt(1);
    }

  virtual void destroy() { }

private:
  optix::float3 m_origin;
  optix::float3 m_u;
  optix::float3 m_v;
  optix::float3 m_w;
};

class ioOrthographicCamera : public ioCamera
{
public:
  ioOrthographicCamera(
    float fromX, float fromY, float fromZ,
    float toX, float toY, float toZ,
    float upX, float upY, float upZ,
    float height, float width
    )
    {
      m_halfHeight = height/2.0f;
      m_halfWidth = width/2.0f;
      m_origin = optix::make_float3(fromX, fromY, fromZ);
      m_w = optix::normalize(
        optix::make_float3(fromX-toX, fromY-toY, fromZ-toZ)
        );
      m_u = optix::normalize(
        optix::cross(make_float3(upX, upY, upZ), m_w)
        );
      m_v = optix::cross(m_w, m_u);
    }

  virtual void init(optix::Context& context)
    {
      context["cameraOrigin"]->set3fv((float*)&m_origin);
      context["cameraU"]->set3fv((float*)&m_u);
      context["cameraV"]->set3fv((float*)&m_v);
      context["cameraW"]->set3fv((float*)&m_w);
      context["cameraHalfHeight"]->setFloat(m_halfHeight);
      context["cameraHalfWidth"]->setFloat(m_halfWidth);
      context["cameraType"]->setInt(2);
    }

  virtual void destroy() { }

private:
  optix::float3 m_origin;
  optix::float3 m_u;
  optix::float3 m_v;
  optix::float3 m_w;
  float m_halfHeight;
  float m_halfWidth;
};
#endif //!IO_CAMERA_H
