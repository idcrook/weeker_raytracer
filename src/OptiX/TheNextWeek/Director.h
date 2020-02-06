#ifndef DIRECTOR_H
#define DIRECTOR_H

// Include before the optix includes
#include "scene/ioScene.h"

#include <optix.h>
#include <optixu/optixpp.h>


class Director
{
public:
    Director(bool verbose) : _verbose(verbose) {}

    void init(unsigned int width, unsigned int height, unsigned int samples);
    void destroy();

    void createScene(unsigned int sceneNumber);
    void renderFrame();
    void printPPM();

private:
    int m_Nx;
    int m_Ny;
    int m_Ns;
    int m_maxRayDepth;

    bool _verbose = false;

    optix::Context m_context;
    optix::Buffer m_outputBuffer;

    optix::Program m_rayGenProgram;
    optix::Program m_missProgram;
    // optix::Program m_exceptionProgram;

    // Scene Objects
    ioScene m_scene;

    void initContext();
    void initOutputBuffer();
    void initRayGenProgram();
    void initMissProgram();
    // void initExceptionProgram();
};

#endif //!DIRECTOR_H
