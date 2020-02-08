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

    static optix::Transform translate(float3& offset, optix::GeometryInstance gi, optix::Context &context){
        optix::Matrix4x4 matrix = translateMatrix(offset);

        optix::GeometryGroup d_world = context->createGeometryGroup();
        //d_world->setAcceleration(context->createAcceleration("Trbvh"));
        d_world->setAcceleration(context->createAcceleration("NoAccel"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform translate(float3& offset, optix::GeometryGroup gg, optix::Context &context){
        optix::Matrix4x4 matrix = translateMatrix(offset);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(gg);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform translate(float3& offset, optix::Transform t, optix::Context &context){
        optix::Matrix4x4 matrix = translateMatrix(offset);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(t);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }


    static optix::Transform rotateAboutPoint(float angleDegrees, float3& point, optix::GeometryInstance gi, optix::Context &context){
        optix::Matrix4x4 matrix = rotateAboutPointMatrix(-angleDegrees * CUDART_PI_F / 180.f, point);

        optix::GeometryGroup d_world = context->createGeometryGroup();
        //d_world->setAcceleration(context->createAcceleration("Trbvh"));
        d_world->setAcceleration(context->createAcceleration("NoAccel"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateX(float angleDegrees, optix::GeometryInstance gi, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixX(-angleDegrees * CUDART_PI_F / 180.f);

        optix::GeometryGroup d_world = context->createGeometryGroup();
        //d_world->setAcceleration(context->createAcceleration("Trbvh"));
        d_world->setAcceleration(context->createAcceleration("NoAccel"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateX(float angleDegrees, optix::GeometryGroup gg, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixX(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(gg);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateX(float angleDegrees, optix::Transform t, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixX(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(t);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateY(float angleDegrees, optix::GeometryInstance gi, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixY(-angleDegrees * CUDART_PI_F / 180.f);

        optix::GeometryGroup d_world = context->createGeometryGroup();
        //d_world->setAcceleration(context->createAcceleration("Trbvh"));
        d_world->setAcceleration(context->createAcceleration("NoAccel"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform translateTransform = context->createTransform();
        translateTransform->setChild(d_world);
        translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return translateTransform;
    }

    static optix::Transform rotateY(float angleDegrees, optix::GeometryGroup gi,  optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixY(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform transf = context->createTransform();
        transf->setChild(gi);
        transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return transf;
    }

    static optix::Transform rotateY(float angleDegrees, optix::Transform t, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixY(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform transf = context->createTransform();
        transf->setChild(t);
        transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return transf;
    }


    static optix::Transform rotateZ(float angleDegrees, optix::GeometryInstance gi, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixZ(-angleDegrees * CUDART_PI_F / 180.f);

        optix::GeometryGroup d_world = context->createGeometryGroup();
        //d_world->setAcceleration(context->createAcceleration("Trbvh"));
        d_world->setAcceleration(context->createAcceleration("NoAccel"));
        d_world->setChildCount(1);
        d_world->setChild(0, gi);

        optix::Transform transf = context->createTransform();
        transf->setChild(d_world);
        transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return transf;
    }

    static optix::Transform rotateZ(float angleDegrees, optix::GeometryGroup gg,  optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixZ(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform rotateTransform = context->createTransform();
        rotateTransform->setChild(gg);
        rotateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return rotateTransform;
    }

    static optix::Transform rotateZ(float angleDegrees, optix::Transform t, optix::Context &context){
        optix::Matrix4x4 matrix = rotateMatrixZ(-angleDegrees * CUDART_PI_F / 180.f);

        optix::Transform rotateTransform = context->createTransform();
        rotateTransform->setChild(t);
        rotateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

        return rotateTransform;
    }

private:

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


        // rotateZ functions
    static optix::Matrix4x4 rotateMatrixZ(float angle){
        float floatM[16] = {
             cosf(angle), sinf(angle), 0.0f, 0.0f,
            -sinf(angle), cosf(angle), 0.0f, 0.0f,
                    0.0f,        0.0f, 1.0f, 0.0f,
                    0.0f,        0.0f, 0.0f, 1.0f
        };
        optix::Matrix4x4 mm(floatM);

        return mm;
    }


};

#endif
