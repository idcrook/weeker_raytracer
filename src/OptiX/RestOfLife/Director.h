#ifndef DIRECTOR_H
#define DIRECTOR_H

// Include before the optix includes
#include "scene/ioScene.h"

#include <optix.h>
#include <optixu/optixpp.h>


class Director
{
public:
    Director(bool verbose, bool debug) : _verbose(verbose), _debug(debug) {}

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

    optix::Context m_context;
    optix::Buffer m_outputBuffer;

    // Scene Objects
    ioScene m_scene;

    void initContext();
    void initOutputBuffer();

    bool _verbose = false;
    bool _debug = false;
};

#endif //!DIRECTOR_H
