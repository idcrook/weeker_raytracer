// std
#include <iostream>

// optix
#include <optix.h>
#include <optixu/optixpp.h>

#define RT_CHECK( func )                                            \
  do                                                                \
    {                                                               \
      RTresult code = func;                                         \
      if( code != RT_SUCCESS ) {                                    \
        const char* errorString;                                    \
        rtContextGetErrorString( context, code, &errorString );     \
        std::cout << "RT_ERROR: " << __FILE__ << ":" << __LINE__    \
                  << " - " << errorString << std::endl;             \
      }                                                             \
    } while(0)

extern "C" const char drawColor_ptx_c[];

int main(int argc, char* argv[])
{
  // Primary RT Objects
  RTcontext context = 0;
  RTprogram rayGenProgram;
  RTbuffer buffer;
  // Cuda program arguments (parameters)
  RTvariable resultBuffer;
  RTvariable color;

  // Parameters
  int width = 1200;
  int height = 800;

  // Create Objects
  //   Context
  RT_CHECK( rtContextCreate( &context ));
  RT_CHECK( rtContextSetRayTypeCount( context, 1 ));
  RT_CHECK( rtContextSetEntryPointCount( context, 1 ));
  //   Buffer
  RT_CHECK( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ));
  RT_CHECK( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT3 ));
  RT_CHECK( rtBufferSetSize2D(buffer, width, height));
  // Connecting Buffer to Context
  RT_CHECK( rtContextDeclareVariable( context, "resultBuffer", &resultBuffer ));
  RT_CHECK( rtVariableSetObject( resultBuffer, buffer ));

  // Set ray generation program
  //   Make has previously turned our cuda into a ptx file: embedded_raygen_program
  //   Create program from ptx
  RT_CHECK( rtProgramCreateFromPTXString(
                                         context,
                                         drawColor_ptx_c,
                                         "drawColor",
                                         &rayGenProgram ));
  //   Program variable plumbing
  RT_CHECK( rtProgramDeclareVariable( rayGenProgram, "color", &color ));
  //   Set out raygen program variable now
  RT_CHECK( rtVariableSet3f( color, 0.462f, 0.725f, 0.05f ));
  //   Hook raygen program and contet
  RT_CHECK( rtContextSetRayGenerationProgram( context, 0, rayGenProgram ));

  // RUN
  RT_CHECK( rtContextValidate( context ));
  RT_CHECK( rtContextLaunch2D( context, 0,      // Entry Point
                               width, height ));

  // Print out to terminal as PPM
  //   Get pointer to output buffer data
  void* bufferData;
  RT_CHECK( rtBufferMap( buffer, &bufferData ));
  //   Print ppm header
  std::cout << "P3\n" << width << " " << height << "\n255\n";
  //   Parse through bufferdata
  for (int j = height - 1; j >= 0;  j--)
    {
      for (int i = 0; i < width; i++)
        {
          float* floatData = ((float*)bufferData) + (3*(width*j + i));
          float r = floatData[0];
          float g = floatData[1];
          float b = floatData[2];
          int ir = int(255.99f * r);
          int ig = int(255.99f * g);
          int ib = int(255.99f * b);
          std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
  RT_CHECK( rtBufferUnmap( buffer ));

  // Explicitly destroy our objects
  RT_CHECK( rtBufferDestroy( buffer ));
  RT_CHECK( rtProgramDestroy( rayGenProgram));
  RT_CHECK( rtContextDestroy( context ));
  // @Writing Variables are not explicitly destroyed
}
