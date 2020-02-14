#ifndef IO_PDF_H
#define IO_PDF_H

#include <optix.h>
#include <optixu/optixpp.h>

#include <random>


extern "C" const char cosine_pdf_ptx_c[];
extern "C" const char rect_pdf_ptx_c[];
//extern "C" const char sphere_pdf_ptx_c[];
extern "C" const char mixture_pdf_ptx_c[];
extern "C" const char mixture_bias_pdf_ptx_c[];


struct ioPdf {
    virtual optix::Program assignGenerate(optix::Context &context) const = 0;
    virtual optix::Program assignValue(optix::Context &context) const = 0;
};


struct ioCosinePDF : public ioPdf{

    virtual optix::Program assignGenerate(optix::Context &context) const override {
        optix::Program generate = context->createProgramFromPTXString(cosine_pdf_ptx_c, "cosineGenerate");

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &context) const override {
        optix::Program value = context->createProgramFromPTXString(cosine_pdf_ptx_c, "cosineValue");

        return value;
    }
};

// FIXME: refactor to have a struct factory for RectAXIS-s
struct ioRectX_PDF : public ioPdf {
ioRectX_PDF(const float aa0, const float aa1, const float bb0, const float bb1, const float kk)
    : a0(aa0), a1(aa1), b0(bb0), b1(bb1), k(kk) {}

    virtual optix::Program assignGenerate(optix::Context &context) const override {
        optix::Program generate = context->createProgramFromPTXString(rect_pdf_ptx_c, "rect_x_generate");

        generate["a0"]->setFloat(a0);
        generate["a1"]->setFloat(a1);
        generate["b0"]->setFloat(b0);
        generate["b1"]->setFloat(b1);
        generate["k"]->setFloat(k);

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &context) const override {
        optix::Program value = context->createProgramFromPTXString(rect_pdf_ptx_c, "rect_x_value");

        value["a0"]->setFloat(a0);
        value["a1"]->setFloat(a1);
        value["b0"]->setFloat(b0);
        value["b1"]->setFloat(b1);
        value["k"]->setFloat(k);

        return value;
    }

    float a0, a1, b0, b1, k;
};


struct ioRectY_PDF : public ioPdf {
ioRectY_PDF(const float aa0, const float aa1, const float bb0, const float bb1, const float kk)
    : a0(aa0), a1(aa1), b0(bb0), b1(bb1), k(kk) {}

    virtual optix::Program assignGenerate(optix::Context &context) const override {
        optix::Program generate = context->createProgramFromPTXString(rect_pdf_ptx_c, "rect_y_generate");

        generate["a0"]->setFloat(a0);
        generate["a1"]->setFloat(a1);
        generate["b0"]->setFloat(b0);
        generate["b1"]->setFloat(b1);
        generate["k"]->setFloat(k);

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &context) const override {
        optix::Program value = context->createProgramFromPTXString(rect_pdf_ptx_c, "rect_y_value");

        value["a0"]->setFloat(a0);
        value["a1"]->setFloat(a1);
        value["b0"]->setFloat(b0);
        value["b1"]->setFloat(b1);
        value["k"]->setFloat(k);

        return value;
    }

    float a0, a1, b0, b1, k;
};

struct ioRectZ_PDF : public ioPdf {
ioRectZ_PDF(const float aa0, const float aa1, const float bb0, const float bb1, const float kk)
                : a0(aa0), a1(aa1), b0(bb0), b1(bb1), k(kk) {}

    virtual optix::Program assignGenerate(optix::Context &context) const override {
        optix::Program generate = context->createProgramFromPTXString(rect_pdf_ptx_c, "rect_z_generate");

        generate["a0"]->setFloat(a0);
        generate["a1"]->setFloat(a1);
        generate["b0"]->setFloat(b0);
        generate["b1"]->setFloat(b1);
        generate["k"]->setFloat(k);

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &context) const override {
        optix::Program value = context->createProgramFromPTXString(rect_pdf_ptx_c, "rect_z_value");

        value["a0"]->setFloat(a0);
        value["a1"]->setFloat(a1);
        value["b0"]->setFloat(b0);
        value["b1"]->setFloat(b1);
        value["k"]->setFloat(k);

        return value;
    }

    float a0, a1, b0, b1, k;
};

struct ioMixturePDF : public ioPdf {
    ioMixturePDF(const ioPdf *p00, const ioPdf *p11) : p0(p00), p1(p11) {}

    virtual optix::Program assignGenerate(optix::Context &context) const override {
        optix::Program generate = context->createProgramFromPTXString(mixture_pdf_ptx_c, "mixture_generate");

        generate["p0_generate"]->setProgramId(p0->assignGenerate(context));
        generate["p1_generate"]->setProgramId(p1->assignGenerate(context));

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &context) const override {
        optix::Program value = context->createProgramFromPTXString(mixture_pdf_ptx_c, "mixture_value");

        value["p0_value"]->setProgramId(p0->assignValue(context));
        value["p1_value"]->setProgramId(p1->assignValue(context));

        return value;
    }

    const ioPdf* p0;
    const ioPdf* p1;
};


struct ioMixtureBiasPDF : public ioPdf {
ioMixtureBiasPDF(const ioPdf *p00, const ioPdf *p11, const float bias00) : p0(p00), p1(p11), bias(bias00) {}

    virtual optix::Program assignGenerate(optix::Context &context) const override {
        optix::Program generate = context->createProgramFromPTXString(mixture_bias_pdf_ptx_c, "mixture_generate");

        generate["p0_generate"]->setProgramId(p0->assignGenerate(context));
        generate["p1_generate"]->setProgramId(p1->assignGenerate(context));

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &context) const override {
        optix::Program value = context->createProgramFromPTXString(mixture_bias_pdf_ptx_c, "mixture_value");

        value["p0_value"]->setProgramId(p0->assignValue(context));
        value["p1_value"]->setProgramId(p1->assignValue(context));
        value["bias"]->setFloat(bias);

        return value;
    }

    const ioPdf* p0;
    const ioPdf* p1;
    const float bias;
};


#endif //IO_PDF_H
