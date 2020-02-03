#ifndef IO_CAMERA_H
#define IO_CAMERA_H

#include <optix.h>
#include <optixu/optixpp.h>

// defines CUDART_PI_F
#include "math_constants.h"

class ioCamera
{
public:
    ioCamera() {}

    ioCamera(float fromX, float fromY, float fromZ,
             float toX, float toY, float toZ,
             float upX, float upY, float upZ,
             float t0 = 0.f, float t1 = 0.f ) {
        m_origin = optix::make_float3(fromX, fromY, fromZ);
        // w = unit_vector(lookfrom - lookat)
        m_w = optix::normalize(
            optix::make_float3(fromX-toX, fromY-toY, fromZ-toZ)
            );
        // u = unit_vector(cross(vup, w))
        m_u = optix::normalize(
            optix::cross(make_float3(upX, upY, upZ), m_w)
            );
        // v = cross(w, u)
        m_v = optix::cross(m_w, m_u);

        // shutter is open between t0 and
        m_time0 = t0;
        m_time1 = t1;
    }

    virtual ~ioCamera() {}

    virtual void destroy() = 0;

    virtual void init(optix::Context& context) {
        context["cameraOrigin"]->set3fv((float*)&m_origin);
        context["cameraU"]->set3fv((float*)&m_u);
        context["cameraV"]->set3fv((float*)&m_v);
        context["cameraW"]->set3fv((float*)&m_w);
        context["cameraTime0"]->setFloat(m_time0);
        context["cameraTime1"]->setFloat(m_time1);
    }

protected:
    float3 m_origin;
    float3 m_u;
    float3 m_v;
    float3 m_w;
    float m_time0;
    float m_time1;
};


class ioPerspectiveCamera : public ioCamera
{
public:
    ioPerspectiveCamera(
        float fromX, float fromY, float fromZ,
        float toX, float toY, float toZ,
        float upX, float upY, float upZ,
        float vFov, float aspect,
        float aperture, float focus_dist,
        float t0 = 0.f, float t1 = 0.f)
        : ioCamera(fromX, fromY, fromZ,
                   toX, toY, toZ,
                   upX, upY, upZ,
                   t0, t1)
        {

            m_lensRadius = aperture / 2.0f;

            // vFov is top to bottom in degrees
            float theta = vFov * CUDART_PI_F/180.0f;
            float halfHeight = tanf(theta/2.0f);
            float halfWidth = aspect * halfHeight;

            m_lowerLeftCorner = m_origin
                -  halfWidth*focus_dist*m_u
                - halfHeight*focus_dist*m_v
                -            focus_dist*m_w;

            m_horizontal = 2.0f*halfWidth *focus_dist*m_u;
            m_vertical =   2.0f*halfHeight*focus_dist*m_v;
        }

    virtual void init(optix::Context& context)
        {
            ioCamera::init(context);
            context["cameraLowerLeftCorner"]->set3fv(&m_lowerLeftCorner.x);
            context["cameraHorizontal"]->set3fv(&m_horizontal.x);
            context["cameraVertical"]->set3fv(&m_vertical.x);
            context["cameraType"]->setInt(0);
        }

    virtual void destroy() { }

private:
    float3 m_lowerLeftCorner;
    float3 m_horizontal;
    float3 m_vertical;
    float m_lensRadius;

};

class ioEnvironmentCamera : public ioCamera
{
public:
    ioEnvironmentCamera(
        float fromX, float fromY, float fromZ,
        float toX, float toY, float toZ,
        float upX, float upY, float upZ,
        float t0 = 0.f, float t1 = 0.f)
        : ioCamera(fromX, fromY, fromZ,
                   toX, toY, toZ,
                   upX, upY, upZ) { }

    virtual void init(optix::Context& context)
        {
            ioCamera::init(context);
            context["cameraType"]->setInt(1);
        }

    virtual void destroy() { }

};

class ioOrthographicCamera : public ioCamera
{
public:
    ioOrthographicCamera(
        float fromX, float fromY, float fromZ,
        float toX, float toY, float toZ,
        float upX, float upY, float upZ,
        float height, float width,
        float t0 = 0.f, float t1 = 0.f)
        : ioCamera(fromX, fromY, fromZ,
                   toX, toY, toZ,
                   upX, upY, upZ)
        {
            float halfHeight = height/2.0f;
            float halfWidth = width/2.0f;
            m_lowerLeftCorner = m_origin
                -  halfWidth*m_u
                - halfHeight*m_v
                -            m_w;

            m_horizontal = 2.0f*halfWidth*m_u;
            m_vertical =   2.0f*halfHeight*m_v;
        }

    virtual void init(optix::Context& context)
        {
            ioCamera::init(context);
            context["cameraLowerLeftCorner"]->set3fv(&m_lowerLeftCorner.x);
            context["cameraHorizontal"]->set3fv(&m_horizontal.x);
            context["cameraVertical"]->set3fv(&m_vertical.x);
            context["cameraType"]->setInt(2);
        }

    virtual void destroy() { }

private:
    float3 m_lowerLeftCorner;
    float3 m_horizontal;
    float3 m_vertical;
};
#endif //!IO_CAMERA_H
