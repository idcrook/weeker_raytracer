
#include <iostream>
#include <stdexcept>
#include <chrono>

// Image I/O
// define STB_IMAGE*_IMPLEMENTATION-s only once (e.g. in .cpp file)
#define STB_IMAGE_IMPLEMENTATION1
// #define STB_IMAGE_WRITE_IMPLEMENTATION // not yet used
#include "../external/rtw_stb_image.h"

// optix
#include <optix.h>
#include <optixu/optixpp.h>

// Executive Director
#include "Director.h"
// Parse command line arguments and options
#include "InputParser.h"

#define Nscene_MAX  (3)
#define Ns_MAX  (1024*10)

int main(int argc, char* argv[])
{
    int exit_code = EXIT_SUCCESS;

    int Nscene = 0;
    int Ns = 1024;

    InputParser cl_input(argc, argv);
    if(cl_input.cmdOptionExists("-h")){
        std::cout << std::endl <<  " HELP - " << argv[0] << std::endl;
        std::cout << R"(
    -s N           Scene Selection number N (N: 0, 1, 2, etc.)
    -n N           Sample each pixel N times (N: 1, 2, etc.)
    -dx Nx         (TBD) Output image width (x dimension)
    -dy Ny         (TBD) Output image height (y dimension)

    -h             This help message.
    -v             (TBD) Verbose
    -g             (TBD) Debug

)";
        std::exit( exit_code );
    }
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

    const std::string &numberOfSamples = cl_input.getCmdOption("-n");
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
        std::cerr << "Invalid scene number: " << sceneNumber << std::endl;
    }

    Director optixSingleton = Director();


    auto start = std::chrono::system_clock::now();
    //optixSingleton.init(1200, 600);
    optixSingleton.init(560, 560, Ns); // cornell box resolution
    std::cerr << "INFO: Number of rays sent per pixel: " << Ns << std::endl;
    std::cerr << "INFO: Scene number selected: " << Nscene << std::endl;
    optixSingleton.createScene(Nscene);

    optixSingleton.renderFrame();
    auto stop = std::chrono::system_clock::now();
    auto time_seconds = std::chrono::duration<float>(stop - start).count();
    std::cerr << "INFO: Took " << time_seconds << " seconds." << std::endl;

    optixSingleton.printPPM();

    optixSingleton.destroy();
}
