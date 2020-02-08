#ifndef IO_TRANSFORM_H
#define IO_TRANSFORM_H

#include <optix.h>
#include <optixu/optixpp.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <math_constants.h>

// translate utility functions
class ioTransform
{
public:

    static optix::Matrix4x4 translateMatrix(float3 offset){
        float floatM[16] = {
            1.0f, 0.0f, 0.0f, offset.x,
            0.0f, 1.0f, 0.0f, offset.y,
            0.0f, 0.0f, 1.0f, offset.z,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        optix::Matrix4x4 mm(floatM);

        return mm;
    }

    static optix::Transform translate(optix::GeometryInstance gi, float3& translate, optix::Context &g_context){
        optix::Matrix4x4 matrix = translateMatrix(translate);

        optix::GeometryGroup d_world = g_context->createGeometryGroup();
        d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform translate(optix::GeometryGroup gi, float3& translate, optix::Context &g_context){
        optix::Matrix4x4 matrix = translateMatrix(translate);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(gi);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform translate(optix::Transform gi, float3 & translate, optix::Context &g_context){
        optix::Matrix4x4 matrix = translateMatrix(translate);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(gi);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }


    // rotateAboutPoint
    static optix::Matrix4x4 rotateAboutPointMatrix(float angle, float3 offset){
        float floatM[16] = {
            cosf(angle), 0.0f, -sinf(angle), offset.x - cosf(angle) * offset.x + sinf(angle) * offset.z,
            0.0f,        1.0f,         0.0f,                                                        0.f,
            sinf(angle), 0.0f,  cosf(angle), offset.z - sinf(angle) * offset.x - cosf(angle) * offset.z,
            0.0f,        0.0f,         0.0f,                                                       1.0f
        };
        optix::Matrix4x4 mm(floatM);

        return mm;
    }

    // it's *really* slow.
    static optix::Transform rotateAboutPoint(optix::GeometryInstance gi, float angle, float3& translate, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateAboutPointMatrix(-angle * CUDART_PI_F / 180.f, translate);

        optix::GeometryGroup d_world = g_context->createGeometryGroup();
        d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    // rotateX functions
    static optix::Matrix4x4 rotateMatrixX(float angle){
        float floatM[16] = {
            1.0f,         0.0f,        0.0f, 0.0f,
            0.0f,  cosf(angle), sinf(angle), 0.0f,
            0.0f, -sinf(angle), cosf(angle), 0.0f,
            0.0f,         0.0f,        0.0f, 1.0f
        };
        optix::Matrix4x4 mm(floatM);

        return mm;
    }

    static optix::Transform rotateX(optix::GeometryInstance gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixX(-angleDegrees * CUDART_PI_F / 180.f);

        optix::GeometryGroup d_world = g_context->createGeometryGroup();
        d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateX(optix::GeometryGroup gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixX(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(gi);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateX(optix::Transform gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixX(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(gi);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    // rotate  Y functions
    static optix::Matrix4x4 rotateMatrixY(float angle){
        float floatM[16] = {
            cosf(angle), 0.0f, -sinf(angle), 0.0f,
            0.0f,  1.0f,         0.0f, 0.0f,
            sinf(angle), 0.0f,  cosf(angle), 0.0f,
            0.0f,  0.0f,         0.0f, 1.0f
        };
        optix::Matrix4x4 mm(floatM);

        return mm;
    }

    static optix::Transform rotateY(optix::GeometryInstance gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixY(-angleDegrees * CUDART_PI_F / 180.f);

        optix::GeometryGroup d_world = g_context->createGeometryGroup();
        d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = g_context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateY(optix::GeometryGroup gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixY(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform transf = g_context->createTransform();
        transf->setChild(gi);
        transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return transf;
    }

    static optix::Transform rotateY(optix::Transform gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixY(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform transf = g_context->createTransform();
        transf->setChild(gi);
        transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return transf;
    }


    // rotateZ functions
    static optix::Matrix4x4 rotateMatrixZ(float angle){
        float floatM[16] = {
            cosf(angle), sinf(angle), 0.0f, 0.0f,
            -sinf(angle), cosf(angle), 0.0f, 0.0f,
            0.0f,       0.0f, 1.0f, 0.0f,
            0.0f,       0.0f, 0.0f, 1.0f
        };
        optix::Matrix4x4 mm(floatM);

        return mm;
    }

    static optix::Transform rotateZ(optix::GeometryInstance gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixZ(-angleDegrees * CUDART_PI_F / 180.f);

        optix::GeometryGroup d_world = g_context->createGeometryGroup();
        d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform transf = g_context->createTransform();
        transf->setChild(d_world);
        transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return transf;
    }

    static optix::Transform rotateZ(optix::GeometryGroup gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixZ(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform rotateTransform = g_context->createTransform();
        rotateTransform->setChild(gi);
        rotateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return rotateTransform;
    }

    static optix::Transform rotateZ(optix::Transform gi, float angleDegrees, optix::Context &g_context){
        optix::Matrix4x4 matrix = rotateMatrixZ(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform rotateTransform = g_context->createTransform();
        rotateTransform->setChild(gi);
        rotateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return rotateTransform;
    }

};

#endif
