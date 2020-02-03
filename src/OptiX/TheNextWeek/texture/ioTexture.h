#ifndef IO_TEXTURE_H
#define IO_TEXTURE_H

#include <random>

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char null_texture_ptx_c[];
extern "C" const char constant_texture_ptx_c[];
extern "C" const char checkered_texture_ptx_c[];
extern "C" const char noise_texture_ptx_c[];

static __inline__ float localRnd() {
 // static std::random_device rd;  //Will be used to obtain a seed for the rand
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd(
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

struct ioTexture {
    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
    virtual optix::Program getProgram(optix::Context &g_context) const = 0;

};

struct ioNullTexture : public ioTexture {

ioNullTexture() : blank(make_float3(0.f, 0.f, 0.f)) {}

    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg =  getProgram(g_context);
        textProg["color"]->setFloat(blank);
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program  getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(null_texture_ptx_c, "sampleTexture");
        return textProg;
    }

    const float3 blank;
};


struct ioConstantTexture : public ioTexture {
    ioConstantTexture(const float3 &c) : color(c) {}

    virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg =  getProgram(g_context);
        textProg["color"]->setFloat(color);
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program  getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(constant_texture_ptx_c, "sampleTexture");
        return textProg;
    }

    const float3 color;
};


struct ioCheckerTexture : public ioTexture {
ioCheckerTexture(const ioTexture *o, const ioTexture *e) : odd(o), even(e) {}

    virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = getProgram(g_context);
        textProg["odd"]->setProgramId(odd->assignTo(gi, g_context));
        textProg["even"]->setProgramId(even->assignTo(gi, g_context));
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(checkered_texture_ptx_c, "sampleTexture");
        return textProg;
    }
    const ioTexture* odd;
    const ioTexture* even;
};

struct ioNoiseTexture : public ioTexture {
    ioNoiseTexture(const float s) : scale(s) {}

    virtual float3 unit_float3(float x, float y, float z) const {
        float l = sqrt(x*x + y*y + z*z);
        return make_float3(x/l, y/l, z/l);
    }

    void permute(int *p) const {
        for (int i = 256 - 1; i > 0; i--) {
		    int target = int(localRnd() * (i + 1));
		    int tmp = p[i];

		    p[i] = p[target];
		    p[target] = tmp;
	    }
    }

    void perlin_generate_perm(optix::Buffer &perm_buffer, optix::Context &g_context) const {
        perm_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 256);
        int *perm_map = static_cast<int*>(perm_buffer->map());

        for (int i = 0; i < 256; i++)
		    perm_map[i] = i;
        permute(perm_map);
        perm_buffer->unmap();
    }

    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Buffer ranvec = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 256);
        float3 *ranvec_map = static_cast<float3*>(ranvec->map());

        for (int i = 0; i < 256; ++i)
            ranvec_map[i] = unit_float3(-1 + 2 * localRnd(), -1 + 2 * localRnd(), -1 + 2 * localRnd());
        ranvec->unmap();

        optix::Buffer perm_x, perm_y, perm_z;
        perlin_generate_perm(perm_x, g_context);
        perlin_generate_perm(perm_y, g_context);
        perlin_generate_perm(perm_z, g_context);

        optix::Program textProg =  getProgram(g_context);
        textProg["ranvec"]->set(ranvec);
        textProg["perm_x"]->set(perm_x);
        textProg["perm_y"]->set(perm_y);
        textProg["perm_z"]->set(perm_z);

        gi["sampleTexture"]->setProgramId(textProg);

        return textProg;
    }

    virtual optix::Program getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(noise_texture_ptx_c, "sampleTexture");
        return textProg;
    };

    const float scale;
};


#endif
