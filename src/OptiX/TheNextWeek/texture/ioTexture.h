#ifndef IO_TEXTURE_H
#define IO_TEXTURE_H

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char constant_texture_ptx_c[];

struct ioTexture {
    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
};


struct ioConstantTexture : public ioTexture {
    ioConstantTexture(const float3 &c) : color(c) {}

    virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(constant_texture_ptx_c, "constantTexture");
        textProg["color"]->setFloat(color);
        gi["constantTexture"]->setProgramId(textProg);
        return textProg;
    }
    const float3 color;
};

#endif
