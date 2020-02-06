#ifndef IO_TEXTURE_H
#define IO_TEXTURE_H

#include "../external/rtw_stb_image.h"

#include <random>
#include <iostream>

#include <optix.h>
#include <optixu/optixpp.h>

extern "C" const char null_texture_ptx_c[];
extern "C" const char constant_texture_ptx_c[];
extern "C" const char checkered_texture_ptx_c[];
extern "C" const char noise_texture_ptx_c[];
extern "C" const char image_texture_ptx_c[];

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


struct ioImageTexture : public ioTexture{
    ioImageTexture(const std::string f) : fileName(f) {}

    optix::TextureSampler loadTexture(optix::Context context, const std::string fileName) const {
        int nx, ny, nn;
        unsigned char *tex_data = stbi_load((char*)fileName.c_str(), &nx, &ny, &nn, 0);
        std::cerr << "INFO: image " << fileName << " loaded: (" << nx << 'x' << ny << ") depth: " << nn << std::endl;
        optix::TextureSampler sampler = context->createTextureSampler();
        sampler->setWrapMode(0, RT_WRAP_REPEAT);
        sampler->setWrapMode(1, RT_WRAP_REPEAT);
        sampler->setWrapMode(2, RT_WRAP_REPEAT);
        sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        sampler->setMaxAnisotropy(1.0f);
        sampler->setMipLevelCount(1u);
        sampler->setArraySize(1u);

        optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny);
        unsigned char * data = static_cast<unsigned char *>(buffer->map());

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int bindex = (j * nx + i) * 4;
                int iindex = ((ny - j - 1) * nx + i) * nn;
                if (false) { // (i==0) {
                    std::cerr << static_cast<unsigned int>(tex_data[iindex + 0]) << ' '
                              << static_cast<unsigned int>(tex_data[iindex + 1]) << ' '
                              << static_cast<unsigned int>(tex_data[iindex + 2]) << '\t' ;
                }

                data[bindex + 0] = tex_data[iindex + 0];
                data[bindex + 1] = tex_data[iindex + 1];
                data[bindex + 2] = tex_data[iindex + 2];

                if(nn == 4)
                    data[bindex + 3] = tex_data[iindex + 3];
                else//3-channel images
                    data[bindex + 3] = (unsigned char)255;
            }
        }

        buffer->unmap();
        sampler->setBuffer(buffer);
        sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        return sampler;
    }

    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg =  getProgram(g_context);
        textProg["data"]->setTextureSampler(loadTexture(g_context, fileName));
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(image_texture_ptx_c, "sampleTexture");
        return textProg;
    }



    const std::string fileName;
};

#endif
