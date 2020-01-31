#ifndef IO_TEXTURE_H
#define IO_TEXTURE_H

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char constant_texture_ptx_c[];
extern "C" const char checkered_texture_ptx_c[];

struct ioTexture {
    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
};


struct ioConstantTexture : public ioTexture {
    ioConstantTexture(const float3 &c) : color(c) {}

    virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(constant_texture_ptx_c, "sampleTexture");
        textProg["color"]->setFloat(color);
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }
    const float3 color;
};


struct ioCheckerTexture : public ioTexture {
ioCheckerTexture(const ioTexture *o, const ioTexture *e) : odd(o), even(e) {}

    virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(checkered_texture_ptx_c, "sampleTexture");
        textProg["odd"]->setProgramId(odd->assignTo(gi, g_context));
        textProg["even"]->setProgramId(even->assignTo(gi, g_context));
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }
    const ioTexture* odd;
    const ioTexture* even;
};


#endif
