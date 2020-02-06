
#include <iostream>
#include <stdexcept>
#include <chrono>

// Image I/O
// define STB_IMAGE*_IMPLEMENTATION-s only once (e.g. in .cpp file)
#define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION // not yet used
#include "../external/rtw_stb_image.h"

// optix
#include <optix.h>
#include <optixu/optixpp.h>

// Executive Director
#include "Director.h"
// Parse command line arguments and options
#include "InputParser.h"

#define Nx_MIN  (320)
#define Ny_MIN  (200)
// set maximum resolution to standard 4K dimensions
#define Nx_MAX  (3840)
#define Ny_MAX  (2160)
#define Nscene_MAX  (3)   // will need to track actuals
#define Ns_MAX  (1024*10)

int main(int argc, char* argv[])
{
    int exit_code = EXIT_SUCCESS;

    // default values
    int Nx = 1200;
    int Ny = 600;
    int Nscene = 0;
    int Ns = 1024;
    bool Qverbose = false;
    bool Qdebug = false;

    InputParser cl_input(argc, argv);
    if(cl_input.cmdOptionExists("-h")){
        std::cerr << std::endl <<  " HELP - " << argv[0] << std::endl;
        std::cerr << R"(
    -s N           Scene Selection number N (N: 0, 1, 2, etc.)
    -ns N          Sample each pixel N times (N: 1, 2, etc.)
    -dx Nx         Output image width (x dimension)
    -dy Ny         Output image height (y dimension)

    -h             This help message.
    -v             Verbose output.
    -g             Debug output.

)";
        std::exit( exit_code );
    }

    if(cl_input.cmdOptionExists("-v"))
        Qverbose = true;

    if(cl_input.cmdOptionExists("-g"))
        Qdebug = true;

    const std::string &sceneNumber = cl_input.getCmdOption("-s");
    try {
        if (!sceneNumber.empty()){
            std::size_t pos;
            int x = std::stoi(sceneNumber, &pos);
            if (x >= Nscene and x <= Nscene_MAX) {
                Nscene = x;
            } else {
                std::cerr << "WARNING: Scene number " << x << " out of range. Maximum scene number: " << Nscene_MAX << " "
                          << "Using a value of " << Nscene <<std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid scene number: " << sceneNumber << std::endl;
    }

    const std::string &numberOfSamples = cl_input.getCmdOption("-ns");
    try {
        if (!numberOfSamples.empty()){
            std::size_t pos;
            int x = std::stoi(numberOfSamples, &pos);
            if ( (x > 0) && (x <= Ns_MAX))  {
                Ns = x;
            } else {
                std::cerr << "WARNING: Number of samples " << x << " is out of range. ";
                if (x > Ns_MAX) {
                    Ns = Ns_MAX;
                }
                std::cerr << "Using a value of " << Ns << std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid number of samples: " << numberOfSamples << std::endl;
    }

    const std::string &dimWidth = cl_input.getCmdOption("-dx");
    try {
        if (!dimWidth.empty()){
            std::size_t pos;
            int x = std::stoi(dimWidth, &pos);
            // std::cerr << pos << std::endl;
            if (x >= Nx_MIN and x <= Nx_MAX) {
                Nx = x;
            } else {
                std::cerr << "WARNING: Width (-dx) " << x << " out of range. ";
                if (x > Nx_MAX) {
                    Nx = Nx_MAX;
                }
                if (x < Nx_MIN) {
                    Nx = Nx_MIN;
                }
                std::cerr << "Using a value of " << Nx <<std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid image width (-dx): " << dimWidth << std::endl;
    }

    const std::string &dimHeight = cl_input.getCmdOption("-dy");
    try {
        if (!dimHeight.empty()){
            std::size_t pos;
            int x = std::stoi(dimHeight, &pos);
            // std::cerr << pos << std::endl;
            if (x >= Ny_MIN and x <= Ny_MAX) {
                Ny = x;
            } else {
                std::cerr << "WARNING: Width (-dy) " << x << " out of range. ";
                if (x > Ny_MAX) {
                    Ny = Ny_MAX;
                }
                if (x < Ny_MIN) {
                    Ny = Ny_MIN;
                }
                std::cerr << "Using a value of " << Nx <<std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid image height (-dy): " << dimHeight << std::endl;
    }

    Director optixSingleton = Director(Qverbose, Qdebug);

    auto start = std::chrono::system_clock::now();
    optixSingleton.init(Nx, Ny, Ns);

    if (Qverbose) {
        std::cerr << "INFO: Output image dimensions: " << Nx << 'x' << Ny << std::endl;
        std::cerr << "INFO: Number of rays sent per pixel: " << Ns << std::endl;
        std::cerr << "INFO: Scene number selected: " << Nscene << std::endl;
    }
    optixSingleton.createScene(Nscene);

    optixSingleton.renderFrame();
    auto stop = std::chrono::system_clock::now();
    auto time_seconds = std::chrono::duration<float>(stop - start).count();
    std::cerr << "INFO: Took " << time_seconds << " seconds." << std::endl;

    optixSingleton.printPPM();

    optixSingleton.destroy();
}
